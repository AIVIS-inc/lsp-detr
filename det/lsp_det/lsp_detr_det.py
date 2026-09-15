"""``LSPDetrDetection``: Swinv2-T backbone + FeatureSampling + LSP STA decoder with box heads.

Registered in the Dome registry; ``forward(images, targets=None)`` returns the Dome detection
contract ``{pred_logits [B,Q,C], pred_boxes [B,Q,4] cxcywh-normalized, aux_outputs [5]}``.
"""

from __future__ import annotations

import math
import os
import warnings
from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from src.core import register  # Dome registry (sys.path bootstrapped by lsp_det/__init__.py)

from .lsp_trunk import FeatureSampling, LSPTransformerDet, relative_to_absolute_pos


def resolve_pretrained_path(path: str) -> str:
    """A relative checkpoint path (configs/include/lsp_swinv2.yml: ``hf-5class/model.safetensors``) is taken from
    the lsp-detr repo root so the yml is machine-independent; absolute paths are used as given."""
    if not os.path.isabs(path):
        from . import LSP_REPO_ROOT
        path = os.path.join(LSP_REPO_ROOT, path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"hf-5class checkpoint not found: {path} (run det/scripts/fetch_hf5class.py)")
    return path

__all__ = ["LSPDetrDetection"]


class _EncoderStub:
    """Parameter-free stub: ``det_engine.evaluate`` reads ``model.encoder.use_defe`` directly.

    Deliberately *not* an ``nn.Module`` so it owns no parameters/state and cannot interact
    with DDP, EMA or the optimizer regexes.
    """

    use_defe = False

    def __repr__(self) -> str:
        return "_EncoderStub(use_defe=False)"


def build_swinv2_backbone(image_size: int = 1536, window_size: int = 16, embed_dim: int = 96,
                          depths: Sequence[int] = (2, 2, 6, 2), num_heads: Sequence[int] = (3, 6, 12, 24),
                          drop_path_rate: float = 0.1, patch_size: int = 4,
                          out_features: Sequence[str] = ("stage1", "stage2", "stage3", "stage4")):
    """Swinv2-Tiny (patch4/window16) built *offline* from config; weights come from the checkpoint.

    ``image_size`` matters: HF Swinv2 fixes window/shift per stage from ``config.image_size``
    (256 -> stage3 window16/shift0, stage4 window8/shift0; 1536 -> window16/shift8 everywhere).
    """
    from transformers import AutoBackbone, Swinv2Config

    cfg = Swinv2Config(
        image_size=int(image_size), patch_size=int(patch_size), embed_dim=int(embed_dim),
        depths=list(depths), num_heads=list(num_heads), window_size=int(window_size),
        drop_path_rate=float(drop_path_rate), out_features=list(out_features),
    )
    return AutoBackbone.from_config(cfg)


@register()
class LSPDetrDetection(nn.Module):
    __share__ = ["num_classes"]

    def __init__(
        self,
        num_classes: int = 2,
        dim: int = 384,
        num_heads: int = 12,
        query_block_size: float = 256.0 / 18.0,
        feature_levels: Sequence[int] = (2, 1, 0, 2, 1, 0),
        self_sta_config: Optional[dict] = None,
        cross_sta_config: Optional[Sequence[dict]] = None,
        # backbone
        backbone_image_size: int = 1536,
        backbone_window_size: int = 16,
        backbone_drop_path_rate: float = 0.1,
        backbone_freeze_at: int = 1,          # freeze patch-embed + Swin stages [0, freeze_at)
        backbone_freeze_patch_embed: bool = True,
        # heads / policy
        wh_prior_px: Sequence[float] = (14.0, 14.0),
        center_mode: str = "strict-local",    # 'strict-local' | 'movable-reference'
        movable_span_cells: float = 3.0,
        feature_sampling_fixed: bool = False,
        # checkpoint
        pretrained: Optional[str] = None,     # hf-5class/model.safetensors
        pretrained_arm: Optional[str] = None,  # defaults to center_mode
        # misc
        expect_imagenet_norm: bool = True,
        input_norm_check: str = "warn",       # 'warn' | 'raise' | 'off'
    ) -> None:
        super().__init__()
        self_sta_config = dict(self_sta_config or {"kernel": 3, "q_tile": 3, "kv_tile": 3})
        cross_sta_config = list(cross_sta_config or (
            {"kernel": 5, "q_tile": 3, "kv_tile": 8},
            {"kernel": 5, "q_tile": 3, "kv_tile": 4},
            {"kernel": 5, "q_tile": 3, "kv_tile": 2},
        ))
        self.num_classes = int(num_classes)
        self.query_block_size = float(query_block_size)
        self.center_mode = center_mode
        self.feature_sampling_fixed = bool(feature_sampling_fixed)
        self.expect_imagenet_norm = bool(expect_imagenet_norm)
        self.input_norm_check = input_norm_check
        self._norm_checked = False

        self.backbone = build_swinv2_backbone(image_size=backbone_image_size, window_size=backbone_window_size,
                                              drop_path_rate=backbone_drop_path_rate)
        _, *feature_channels, neck = self.backbone.num_features  # [96,192,384], 768
        self.feature_sampling = FeatureSampling(neck, dim)
        # trunk attribute MUST be named `decoder` (optimizer regexes; §5.1)
        self.decoder = LSPTransformerDet(
            dim=dim, num_heads=num_heads, num_classes=self.num_classes, query_block_size=self.query_block_size,
            feature_levels=feature_levels, feature_channels=feature_channels,
            self_sta_config=self_sta_config, cross_sta_config=cross_sta_config,
            wh_prior_px=wh_prior_px, center_mode=center_mode, movable_span_cells=movable_span_cells,
        )
        # det_engine.evaluate() reads model.encoder.use_defe (parameter-free stub, §5.1)
        self.encoder = _EncoderStub()

        self.pretrained_report = None
        if pretrained:
            from .checkpoint import load_hf5class_checkpoint
            pretrained = resolve_pretrained_path(pretrained)
            self.pretrained_report = load_hf5class_checkpoint(self, pretrained, arm=pretrained_arm or center_mode)

        self._freeze_backbone(backbone_freeze_at, backbone_freeze_patch_embed)

    # ------------------------------------------------------------------ freeze policy
    def _freeze_backbone(self, freeze_at: int, freeze_patch_embed: bool) -> None:
        n = 0
        if freeze_patch_embed:
            for p in self.backbone.embeddings.parameters():
                p.requires_grad_(False); n += 1
        for i in range(max(0, int(freeze_at))):
            for p in self.backbone.encoder.layers[i].parameters():
                p.requires_grad_(False); n += 1
        self.frozen_backbone_tensors = n

    # ------------------------------------------------------------------ forward
    def _check_input_norm(self, x: Tensor) -> None:
        if self._norm_checked or self.input_norm_check == "off" or not self.expect_imagenet_norm:
            return
        self._norm_checked = True
        with torch.no_grad():
            lo, hi = float(x.min()), float(x.max())
        if lo >= 0.0 and hi <= 1.0:
            msg = (f"LSPDetrDetection expects ImageNet-normalized input but the first batch lies in "
                   f"[{lo:.3f},{hi:.3f}] (looks like /255-only). Check the transform pipeline (§6.2).")
            if self.input_norm_check == "raise":
                raise RuntimeError(msg)
            warnings.warn(msg)

    def forward(self, x: Tensor, targets=None):
        self._check_input_norm(x)
        b, _, h, w = x.shape
        *features, neck = self.backbone(x).feature_maps
        qh, qw = math.ceil(h / self.query_block_size), math.ceil(w / self.query_block_size)
        ref_points = torch.zeros(b, qh, qw, 2, dtype=torch.float32, device=neck.device)
        pts = relative_to_absolute_pos(ref_points, self.query_block_size, self.query_block_size, span=self.decoder.center_span)
        if self.feature_sampling_fixed:
            pts = pts / torch.tensor([w, h], dtype=pts.dtype, device=pts.device)
        tgt = self.feature_sampling(pts, neck)
        return self.decoder(tgt, ref_points, features, h, w)
