"""LSP-DETR trunk ported from ``hf-5class/modeling.py`` (RationAI/LSP-DETR HF snapshot).

Kept 1:1 with the snapshot so that checkpoint keys/shapes line up:
  CayleySTRING (pe.freqs, pe.parametrizations.S.original), STAttention (q/kv/wo), Layer
  (self_attention/self_attention_norm/cross_attention/cross_attention_norm/ffn/ffn_norm),
  FeedForward (w1/w2/w3), MLP (Sequential 0/2/4), FeatureSampling (reduction/norm),
  point_head, class_head.

Intentional deviations from the snapshot (FINETUNE_STRATEGY.md §5):
  * ``CayleySTRING.P`` is no longer a ``cached_property``: the eval branch recomputes
    (I-S)(I+S)^-1 on every call so EMA/validation always sees the current S (§5.2).
  * ``flex_attention`` is compiled statically (``dynamic=False``); the shapes are fixed
    (deviation ledger #9).
  * The STA block mask cache is keyed on the tensor device (needed for DDP / CPU tests);
    behaviour is otherwise identical.
  * ``radial_distances_head`` (64-d) is replaced by ``wh_head`` (2-d log-wh residual head),
    the class head is ``Linear(dim, num_classes)`` (sigmoid VFL, no explicit background),
    and the decoder emits Dome-style ``pred_logits / pred_boxes`` (§4.1).
  * Centre policy switch: ``strict-local`` (snapshot behaviour, one sigmoid cell) or
    ``movable-reference`` (query centre may move over ``movable_span_cells`` cells) (§4.2).
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention as _flex_attention
from torch.nn.utils import parametrize

# Static shapes only (1536 crops -> 4 attention geometries + eval); dynamic=True is pure loss.
# Raise the recompile budget so parity/overfit smoke tests at other resolutions do not fall
# back to eager silently.
torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
flex_attention = torch.compile(_flex_attention, dynamic=False)


# ----------------------------------------------------------------------------- Cayley-STRING
def init_freqs(head_dim: int, num_heads: int, pos_dim: int, theta: float) -> Tensor:
    """Taken from https://github.com/naver-ai/rope-vit/blob/main/self-attn/rope_self_attn.py."""
    freqs_x = []
    freqs_y = []
    freqs = 1 / (theta ** (torch.arange(0, head_dim, 2 * pos_dim).float() / head_dim))
    for _ in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi
        fx = torch.cat([freqs * torch.cos(angles), freqs * torch.cos(torch.pi / 2 + angles)], dim=-1)
        fy = torch.cat([freqs * torch.sin(angles), freqs * torch.sin(torch.pi / 2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    return torch.stack([freqs_x, freqs_y], dim=0)


class Skew(nn.Module):
    """Skew-symmetric matrix parameterization."""

    def forward(self, x: Tensor) -> Tensor:
        a = x.triu(1)
        return a - a.transpose(-1, -2)

    def right_inverse(self, x: Tensor) -> Tensor:
        return x.triu(1)


class CayleySTRING(nn.Module):
    """Cayley-STRING positional encoding (RoPE-mixed followed by learnable orthogonal P).

    P = (I - S)(I + S)^-1 with S skew-symmetric. NOTE: unlike the HF snapshot, ``P`` is a
    plain method (no ``cached_property``): a cached P went stale after the first eval pass
    while S kept training (EMA validation would silently use the first-epoch P).
    """

    def __init__(self, dim: int, num_heads: int, pos_dim: int = 2, theta: float = 100.0) -> None:
        super().__init__()
        assert dim % num_heads == 0, "Dimension must be divisible by num_heads."
        head_dim = dim // num_heads
        self.freqs = nn.Parameter(init_freqs(head_dim, num_heads, pos_dim, theta))
        self.S = nn.Parameter(torch.zeros(head_dim, head_dim))
        parametrize.register_parametrization(self, "S", Skew())
        self.register_buffer("I", torch.eye(head_dim), persistent=False)
        self.init_weights()

    def init_weights(self) -> None:
        self.S = nn.init.kaiming_uniform_(self.S, a=math.sqrt(5))

    def P(self) -> Tensor:
        """Recomputed on every call (32x32 inverse; negligible)."""
        i_plus_s_inv = torch.linalg.inv(self.I + self.S)
        return torch.matmul(self.I - self.S, i_plus_s_inv)

    @parametrize.cached()
    @torch.autocast("cuda", enabled=False)
    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        """x: [b, h, n, d]; positions: [b, n, pos_dim]."""
        if self.training:
            # linalg.solve for numerical stability during training (as in the snapshot).
            y = torch.linalg.solve(self.I + self.S, rearrange(x.float(), "b h n d -> (b h) d n"))
            px = torch.matmul(self.I - self.S, y)
            px = rearrange(px, "(b h) d n -> b h n d", b=x.size(0))
        else:
            px = x.float() @ self.P().T
        px = px.contiguous()

        # RoPE-Mixed: angles[b,h,n,c] = pos_x * freqs_x + pos_y * freqs_y.
        # Computed elementwise on purpose (the snapshot uses einsum == a cuBLAS GEMM): this machine forces
        # TF32 for cuBLAS (TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1), which rounds pixel positions to a 10-bit
        # mantissa -> ~1.4 rad angle error at 1536 px. Elementwise mul/add stays exact float32 under any
        # TF32 policy; mathematically identical to the snapshot.
        pos = positions.float()
        angles = (pos[:, None, :, 0, None] * self.freqs[0][None, :, None, :]
                  + pos[:, None, :, 1, None] * self.freqs[1][None, :, None, :])
        freqs_cis = torch.polar(torch.ones_like(angles), angles)
        px_ = torch.view_as_complex(rearrange(px, "... (d two) -> ... d two", two=2))
        out = rearrange(torch.view_as_real(px_ * freqs_cis), "... d two -> ... (d two)")
        return out.type_as(x)


# ----------------------------------------------------------------------------- small blocks
class MLP(nn.Sequential):
    """Very simple multi-layer perceptron (Linear/GELU/[Dropout] ... Linear)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int,
                 act_layer: type[nn.Module] = nn.GELU, dropout: float = 0.0) -> None:
        assert num_layers > 1
        layers = []
        h = [hidden_dim] * (num_layers - 1)
        for n, k in zip([input_dim, *h], h, strict=False):
            layers.append(nn.Linear(n, k))
            layers.append(act_layer())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        super().__init__(*layers)


class FeedForward(nn.Module):
    """SwiGLU FFN (llama4 style)."""

    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256) -> None:
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ----------------------------------------------------------------------------- STA mask
def generate_sta_mask(q_canvas_w: int, kv_canvas_hw: tuple[int, int], kernel: int, q_tile: int, kv_tile: int):
    q_canvas_tile_w = q_canvas_w // q_tile
    kv_canvas_tile_h = kv_canvas_hw[0] // kv_tile
    kv_canvas_tile_w = kv_canvas_hw[1] // kv_tile

    def q_tile_rescale(x: Tensor):
        # round(x * (kv_canvas_tile_w - 1) / (q_canvas_tile_w - 1))
        scale_numerator = kv_canvas_tile_w - 1
        scale_denominator = q_canvas_tile_w - 1
        return (x * scale_numerator + scale_denominator // 2) // scale_denominator

    def get_tile_xy(idx: Tensor, tile_size: int, canvas_tile_w: int):
        tile_id = idx // (tile_size * tile_size)
        tile_x = tile_id % canvas_tile_w
        tile_y = tile_id // canvas_tile_w
        return tile_x, tile_y

    def sta_mask_2d(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        q_x_tile, q_y_tile = get_tile_xy(q_idx, q_tile, q_canvas_tile_w)
        kv_x_tile, kv_y_tile = get_tile_xy(kv_idx, kv_tile, kv_canvas_tile_w)
        q_x_tile = q_tile_rescale(q_x_tile)
        q_y_tile = q_tile_rescale(q_y_tile)
        center_x = q_x_tile.clamp(kernel // 2, (kv_canvas_tile_w - 1) - kernel // 2)
        center_y = q_y_tile.clamp(kernel // 2, (kv_canvas_tile_h - 1) - kernel // 2)
        x_mask = torch.abs(center_x - kv_x_tile) <= kernel // 2
        y_mask = torch.abs(center_y - kv_y_tile) <= kernel // 2
        return x_mask & y_mask

    return sta_mask_2d


@lru_cache(maxsize=None)
def create_sta_block_mask(q_len: int, kv_len: int, q_width: int, kv_width: int, kernel: int,
                          q_tile: int, kv_tile: int, device: str) -> BlockMask:
    return create_block_mask(
        generate_sta_mask(q_width, (kv_len // kv_width, kv_width), kernel, q_tile, kv_tile),
        B=None, H=None, device=device, Q_LEN=q_len, KV_LEN=kv_len, _compile=True,
    )


# ----------------------------------------------------------------------------- coordinates
@torch.autocast("cuda", enabled=False)
def relative_to_absolute_pos(pos: Tensor, step_x: float, step_y: float, span: float = 1.0) -> Tensor:
    """Map per-cell logits ``pos`` [b, h, w, 2] to absolute coordinates.

    span == 1 (snapshot / strict-local): abs = sigmoid(pos) * step + anchor           (inside own cell)
    span == R  (movable-reference):     abs = (sigmoid(pos) * R - (R - 1) / 2) * step + anchor
                                        i.e. the centre may move over R cells centred on its own cell;
                                        sigmoid(0)=0.5 still maps to the cell centre for every R.
    """
    pos = pos.sigmoid()
    if span != 1.0:
        pos = pos * span - (span - 1.0) / 2.0
    h, w = pos.shape[1:3]
    anchor_x = torch.arange(w, dtype=torch.float32, device=pos.device) * step_x
    anchor_y = torch.arange(h, dtype=torch.float32, device=pos.device) * step_y
    absolute_x = pos[..., 0] * step_x + anchor_x
    absolute_y = pos[..., 1] * step_y + anchor_y.unsqueeze(1)
    return torch.stack((absolute_x, absolute_y), dim=-1)


# ----------------------------------------------------------------------------- attention
class STAttention(nn.Module):
    def __init__(self, dim: int, src_dim: int, num_heads: int, kernel: int, q_tile: int, kv_tile: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.kernel = kernel
        self.q_tile = q_tile
        self.kv_tile = kv_tile
        self.pe = CayleySTRING(dim, num_heads)
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(src_dim, dim * 2, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def maybe_pad(self, x: Tensor, tile: int) -> Tensor:
        h, w = x.shape[1:3]
        pad_right = (tile - w % tile) % tile
        pad_bottom = (tile - h % tile) % tile
        return F.pad(x, (0, 0, 0, pad_right, 0, pad_bottom))

    def tile(self, x: Tensor, height: int, tile: int) -> tuple[Tensor, int, int]:
        x = rearrange(x, "b head (h w) dim -> b h w (head dim)", h=height)
        x = self.maybe_pad(x, tile)
        h, w = x.shape[1:3]
        x = rearrange(x, "b (n_h ts_h) (n_w ts_w) (h d) -> b h (n_h n_w ts_h ts_w) d",
                      ts_h=tile, ts_w=tile, h=self.num_heads)
        return x, h, w

    def forward(self, tgt: Tensor, src: Tensor, q_coords: Tensor, k_coords: Tensor) -> Tensor:
        h, w = tgt.shape[1:3]
        q = rearrange(self.q(tgt), "b h w (head d) -> b head (h w) d", head=self.num_heads)
        k, v = rearrange(self.kv(src), "b h w (two head d) -> two b head (h w) d", two=2, head=self.num_heads)

        q = self.pe(q, q_coords)
        k = self.pe(k, k_coords)

        q, q_h, q_w = self.tile(q, h, self.q_tile)
        k, _, kv_w = self.tile(k, src.shape[1], self.kv_tile)
        v, _, _ = self.tile(v, src.shape[1], self.kv_tile)

        block_mask = create_sta_block_mask(
            q_len=q.shape[2], kv_len=k.shape[2], q_width=q_w, kv_width=kv_w,
            kernel=self.kernel, q_tile=self.q_tile, kv_tile=self.kv_tile, device=str(q.device),
        )
        x = flex_attention(q, k, v, block_mask=block_mask)

        x = rearrange(x, "b h (n_h n_w ts_h ts_w) d -> b (n_h ts_h) (n_w ts_w) (h d)",
                      n_h=q_h // self.q_tile, n_w=q_w // self.q_tile, ts_h=self.q_tile, ts_w=self.q_tile)
        x = x[:, :h, :w, :].contiguous()
        return self.wo(x)


class Layer(nn.Module):
    def __init__(self, dim: int, src_dim: int, num_heads: int, self_sta_config: dict, cross_sta_config: dict) -> None:
        super().__init__()
        self.self_attention = STAttention(dim, dim, num_heads, kernel=self_sta_config["kernel"],
                                          q_tile=self_sta_config["q_tile"], kv_tile=self_sta_config["kv_tile"])
        self.self_attention_norm = nn.LayerNorm(dim)
        self.cross_attention = STAttention(dim, src_dim, num_heads, kernel=cross_sta_config["kernel"],
                                           q_tile=cross_sta_config["q_tile"], kv_tile=cross_sta_config["kv_tile"])
        self.cross_attention_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dim * 4)
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, tgt: Tensor, src: Tensor, tgt_coords: Tensor, src_coords: Tensor) -> Tensor:
        x = self.self_attention(tgt, tgt, tgt_coords, tgt_coords)
        tgt = self.self_attention_norm(tgt + x)
        x = self.cross_attention(tgt, src, tgt_coords, src_coords)
        tgt = self.cross_attention_norm(tgt + x)
        return self.ffn_norm(tgt + self.ffn(tgt))


class FeatureSampling(nn.Module):
    """1x1 reduction of the neck feature, sampled at the query reference points, then LayerNorm.

    NOTE (§4.3): the snapshot passes *pixel* coordinates to ``grid_sample`` (which expects
    normalized [0,1] before the ``*2-1``), so the sampled query init is effectively a learned
    constant. This is preserved by default for checkpoint parity; ``fixed=True`` normalizes the
    points by (W, H) first (``feature-sampling-fixed`` ablation).
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.reduction = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, points: Tensor, feature: Tensor) -> Tensor:
        x = F.grid_sample(self.reduction(feature), points * 2 - 1, align_corners=False)
        return self.norm(rearrange(x, "b c h w -> b h w c"))


# ----------------------------------------------------------------------------- decoder trunk
class LSPTransformerDet(nn.Module):
    """6-layer STA decoder + point/log-wh/class heads emitting Dome-style detection outputs.

    Attribute names are kept identical to the snapshot's ``LSPTransformer`` where the weights
    are loaded (``layers``, ``point_head``, ``class_head``); the new head is ``wh_head``.
    """

    def __init__(self, dim: int, num_heads: int, num_classes: int, query_block_size: float,
                 feature_levels: Sequence[int], feature_channels: Sequence[int],
                 self_sta_config: dict, cross_sta_config: Sequence[dict],
                 wh_prior_px: Sequence[float], center_mode: str = "strict-local",
                 movable_span_cells: float = 3.0, log_wh_clamp: Sequence[float] = (-6.0, 9.0)) -> None:
        super().__init__()
        assert center_mode in ("strict-local", "movable-reference"), center_mode
        self.query_block_size = float(query_block_size)
        self.feature_levels = list(feature_levels)
        self.num_classes = int(num_classes)
        self.center_mode = center_mode
        self.center_span = 1.0 if center_mode == "strict-local" else float(movable_span_cells)
        assert len(wh_prior_px) == 2 and all(v > 0 for v in wh_prior_px), wh_prior_px
        self.register_buffer("log_wh_prior", torch.log(torch.tensor(list(wh_prior_px), dtype=torch.float32)), persistent=False)
        self.log_wh_clamp = tuple(float(v) for v in log_wh_clamp)

        self.layers = nn.ModuleList()
        for level in self.feature_levels:
            self.layers.append(Layer(dim=dim, src_dim=feature_channels[level], num_heads=num_heads,
                                     self_sta_config=self_sta_config, cross_sta_config=cross_sta_config[level]))

        # output heads (class head is shared across layers, as in the snapshot)
        self.class_head = nn.Linear(dim, self.num_classes)
        self.point_head = nn.ModuleList(MLP(dim, dim, 2, 3) for _ in self.feature_levels)
        self.wh_head = nn.ModuleList(MLP(dim, dim, 2, 3) for _ in self.feature_levels)
        self.init_weights()

    def init_weights(self) -> None:
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.class_head.bias, bias_value)
        for head in self.point_head:
            nn.init.constant_(head[-1].weight, 0)
            nn.init.constant_(head[-1].bias, 0)
        for head in self.wh_head:
            nn.init.constant_(head[-1].weight, 0)
            nn.init.constant_(head[-1].bias, 0)

    # -- coordinate helpers -----------------------------------------------------------------
    def query_pixel_coords(self, ref_points: Tensor) -> Tensor:
        """Absolute pixel coordinates of query centres [b, h*w, 2] (RoPE positions)."""
        return relative_to_absolute_pos(ref_points, self.query_block_size, self.query_block_size,
                                        span=self.center_span).flatten(1, 2)

    def query_norm_center(self, ref_points: Tensor, height: int, width: int) -> Tensor:
        """Normalized (cx, cy) in [0,1] image units [b, h*w, 2]."""
        return relative_to_absolute_pos(ref_points, self.query_block_size / width, self.query_block_size / height,
                                        span=self.center_span).flatten(1, 2)

    def _boxes(self, center: Tensor, log_wh: Tensor, height: int, width: int) -> Tensor:
        log_wh = log_wh.clamp(*self.log_wh_clamp)
        wh = torch.exp(log_wh)  # pixel space
        wh = wh / torch.tensor([width, height], dtype=wh.dtype, device=wh.device)
        return torch.cat([center, wh], dim=-1)

    def forward(self, tgt: Tensor, ref_points: Tensor, features: List[Tensor], height: int, width: int) -> Dict:
        src, src_coords = [], []
        for feature in features:
            b, _, h, w = feature.shape
            coords = torch.zeros(b, h, w, 2, dtype=torch.float32, device=feature.device)
            coords = relative_to_absolute_pos(coords, step_x=math.ceil(width / w), step_y=math.ceil(height / h))
            src.append(rearrange(feature, "b c h w -> b h w c"))  # SwinV2 outputs are already normalized
            src_coords.append(rearrange(coords, "b h w pos -> b (h w) pos"))

        log_wh = self.log_wh_prior.to(tgt.dtype).expand(*tgt.shape[:3], 2).clone()

        logits_list: List[Tensor] = []
        center_list: List[Tensor] = []
        log_wh_list: List[Tensor] = []

        # look forward twice
        new_ref_points = ref_points.clone()
        new_log_wh = log_wh.clone()

        for i, layer in enumerate(self.layers):
            level = self.feature_levels[i]
            tgt = layer(tgt=tgt, src=src[level], tgt_coords=self.query_pixel_coords(ref_points), src_coords=src_coords[level])

            delta_point = self.point_head[i](tgt)
            delta_wh = self.wh_head[i](tgt)
            logits = self.class_head(tgt)

            center_list.append(self.query_norm_center(new_ref_points + delta_point, height, width))
            log_wh_list.append(torch.flatten(new_log_wh + delta_wh, 1, 2))
            logits_list.append(logits.flatten(1, 2))

            new_ref_points = ref_points + delta_point
            new_log_wh = log_wh + delta_wh
            ref_points = new_ref_points.detach()
            log_wh = new_log_wh.detach()

        boxes_list = [self._boxes(c, s, height, width) for c, s in zip(center_list, log_wh_list)]

        out = {
            "pred_logits": logits_list[-1],
            "pred_boxes": boxes_list[-1],
            "aux_outputs": [{"pred_logits": a, "pred_boxes": b} for a, b in zip(logits_list[:-1], boxes_list[:-1])],
            # extras (not consumed by Dome criterion/postprocessor)
            "pred_center": center_list[-1],
            "pred_log_wh": log_wh_list[-1],
            "embeddings": tgt.flatten(1, 2),
        }
        return out
