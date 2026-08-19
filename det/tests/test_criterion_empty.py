"""§5.3 / §6.1: LSPCriterion handles empty targets and all-empty batches; loss keys are final+5 aux only."""
import math

import pytest
import torch

from src.core import create, GLOBAL_CONFIG
from src.zoo.dome.matcher import HungarianMatcher

from lsp_det.lsp_criterion import LSPCriterion

Q, C = 324, 2  # 18x18 queries at 256 input


def _criterion(use_uni_set=True):
    matcher = HungarianMatcher(weight_dict={"cost_class": 3, "cost_bbox": 3, "cost_giou": 1}, use_focal_loss=True, alpha=0.25, gamma=2.0)
    return LSPCriterion(matcher=matcher, weight_dict={"loss_vfl": 3, "loss_bbox": 3, "loss_giou": 1},
                        losses=["vfl", "boxes"], alpha=0.75, gamma=2.0, num_classes=C, use_uni_set=use_uni_set)


def _outputs(B, requires_grad=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    def one():
        logits = torch.randn(B, Q, C, generator=g) - 4.0
        boxes = torch.cat([torch.rand(B, Q, 2, generator=g), torch.rand(B, Q, 2, generator=g) * 0.05 + 0.005], -1)
        return logits.requires_grad_(requires_grad), boxes.requires_grad_(requires_grad)
    l, b = one()
    aux = []
    for _ in range(5):
        la, ba = one(); aux.append({"pred_logits": la, "pred_boxes": ba})
    return {"pred_logits": l, "pred_boxes": b, "aux_outputs": aux}


def _target(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"labels": torch.randint(0, C, (n,), generator=g),
            "boxes": torch.cat([torch.rand(n, 2, generator=g), torch.rand(n, 2, generator=g) * 0.05 + 0.005], -1)}


def _empty_target():
    return {"labels": torch.zeros((0,), dtype=torch.int64), "boxes": torch.zeros((0, 4), dtype=torch.float32)}


def test_loss_keys_final_plus_5_aux_only():
    crit = _criterion()
    out = _outputs(2)
    losses = crit(out, [_target(10), _target(3)], epoch=0, step=0)
    keys = sorted(losses)
    expect = sorted(["loss_vfl", "loss_bbox", "loss_giou"] + [f"{k}_aux_{i}" for i in range(5) for k in ("loss_vfl", "loss_bbox", "loss_giou")])
    assert keys == expect, keys
    assert all(torch.isfinite(v) for v in losses.values())
    total = sum(losses.values()); total.backward()
    assert torch.isfinite(out["pred_logits"].grad).all() and torch.isfinite(out["pred_boxes"].grad).all()


def test_mixed_batch_one_empty_target():
    crit = _criterion()
    out = _outputs(2)
    losses = crit(out, [_target(7), _empty_target()], epoch=0, step=0)
    assert all(torch.isfinite(v) for v in losses.values())
    sum(losses.values()).backward()
    assert torch.isfinite(out["pred_logits"].grad).all()


def test_all_empty_batch_regression_zero_vfl_finite_positive():
    crit = _criterion()
    out = _outputs(2)
    losses = crit(out, [_empty_target(), _empty_target()], epoch=0, step=0)
    for k, v in losses.items():
        assert torch.isfinite(v), k
        if "bbox" in k or "giou" in k:
            assert float(v) == 0.0, (k, float(v))
        else:  # vfl: background-negative loss on all queries, finite and > 0
            assert float(v) > 0.0, (k, float(v))
    sum(losses.values()).backward()
    assert torch.isfinite(out["pred_logits"].grad).all()
    # boxes get no gradient from an all-empty batch (regression terms are exactly 0)
    assert out["pred_boxes"].grad is None or float(out["pred_boxes"].grad.abs().sum()) == 0.0


def test_num_boxes_normalisation_clamps_to_one_on_empty():
    crit = _criterion()
    out = _outputs(1)
    losses = crit(out, [_empty_target()])
    # with num_boxes clamped to 1, VFL = mean over queries * Q / 1 -> equals plain sum over queries of BCE-weighted terms
    assert float(losses["loss_vfl"]) > 0


def test_registry_creation_from_dome_style_config():
    """LSPCriterion is creatable through the Dome registry with the recipe values."""
    cfg = {
        "num_classes": 2, "use_focal_loss": True,
        "LSPCriterion": {"weight_dict": {"loss_vfl": 3, "loss_bbox": 3, "loss_giou": 1}, "losses": ["vfl", "boxes"],
                          "alpha": 0.75, "gamma": 2.0,
                          "matcher": {"type": "HungarianMatcher", "weight_dict": {"cost_class": 3, "cost_bbox": 3, "cost_giou": 1}, "alpha": 0.25, "gamma": 2.0}},
    }
    from src.core.yaml_utils import merge_config
    gcfg = merge_config(cfg, inplace=False, overwrite=False)
    crit = create("LSPCriterion", gcfg)
    assert isinstance(crit, LSPCriterion) and crit.num_classes == 2 and crit.matcher.use_focal_loss is True
    assert crit.alpha == 0.75 and crit.matcher.alpha == 0.25
