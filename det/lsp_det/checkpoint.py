"""Controlled loading of ``hf-5class/model.safetensors`` into ``LSPDetrDetection`` (§3.1).

Policy (strict-local arm):
  load    backbone.*, feature_sampling.*, decode_head.layers.* -> decoder.layers.*,
          decode_head.point_head.* -> decoder.point_head.*
  remap   decode_head.radial_distances_head.{i}.{0,2}.* -> decoder.wh_head.{i}.{0,2}.*   (hidden MLP layers)
  drop    decode_head.radial_distances_head.{i}.4.*  (64-d radial output)  -> wh_head.{i}.4 stays zero-init
          decode_head.class_head.*                   (6-class K+1 head)     -> class_head stays new (bias prior 0.01)
movable-reference arm additionally drops decode_head.point_head.{i}.4.* (semantics change) and
zero-initialises decoder.point_head.{i}.4.

``load_state_dict(strict=False)`` is never used blindly: the missing / dropped key sets are asserted
to be *exactly* the expected ones and every loaded tensor is shape-checked.
"""

from __future__ import annotations

import re
from typing import Dict, Set

import torch
import torch.nn as nn

__all__ = ["load_hf5class_checkpoint", "remap_hf5class_state_dict", "expected_key_sets"]

_RADIAL_HIDDEN = re.compile(r"^decode_head\.radial_distances_head\.(\d+)\.(0|2)\.(weight|bias)$")
_RADIAL_OUT = re.compile(r"^decode_head\.radial_distances_head\.(\d+)\.4\.(weight|bias)$")
_POINT_OUT = re.compile(r"^decode_head\.point_head\.(\d+)\.4\.(weight|bias)$")
_CLASS = re.compile(r"^decode_head\.class_head\.(weight|bias)$")


def remap_hf5class_state_dict(sd: Dict[str, torch.Tensor], arm: str):
    """Return (remapped_state_dict, dropped_source_keys)."""
    assert arm in ("strict-local", "movable-reference"), arm
    out, dropped = {}, set()
    for k, v in sd.items():
        if _CLASS.match(k) or _RADIAL_OUT.match(k):
            dropped.add(k); continue
        if arm == "movable-reference" and _POINT_OUT.match(k):
            dropped.add(k); continue
        m = _RADIAL_HIDDEN.match(k)
        if m:
            out[f"decoder.wh_head.{m.group(1)}.{m.group(2)}.{m.group(3)}"] = v
            continue
        if k.startswith("decode_head."):
            out["decoder." + k[len("decode_head."):]] = v
        else:
            out[k] = v  # backbone.*, feature_sampling.*
    return out, dropped


def expected_key_sets(model: nn.Module, arm: str):
    """(expected_missing_model_keys, expected_dropped_source_keys) for the given arm."""
    n_layers = len(model.decoder.layers)
    missing: Set[str] = {"decoder.class_head.weight", "decoder.class_head.bias"}
    dropped: Set[str] = {"decode_head.class_head.weight", "decode_head.class_head.bias"}
    for i in range(n_layers):
        for p in ("weight", "bias"):
            missing.add(f"decoder.wh_head.{i}.4.{p}")
            dropped.add(f"decode_head.radial_distances_head.{i}.4.{p}")
            if arm == "movable-reference":
                missing.add(f"decoder.point_head.{i}.4.{p}")
                dropped.add(f"decode_head.point_head.{i}.4.{p}")
    return missing, dropped


def load_hf5class_checkpoint(model: nn.Module, path: str, arm: str = "strict-local", verbose: bool = True) -> dict:
    from safetensors.torch import load_file

    sd = load_file(path)
    remapped, dropped = remap_hf5class_state_dict(sd, arm)
    exp_missing, exp_dropped = expected_key_sets(model, arm)

    model_sd = model.state_dict()
    # shape check on every key we intend to load
    bad = [(k, tuple(v.shape), tuple(model_sd[k].shape)) for k, v in remapped.items() if k in model_sd and tuple(v.shape) != tuple(model_sd[k].shape)]
    assert not bad, f"shape mismatch on remapped keys: {bad[:10]}"

    res = model.load_state_dict(remapped, strict=False)
    missing, unexpected = set(res.missing_keys), set(res.unexpected_keys)

    assert unexpected == set(), f"unexpected keys after remap (should be none): {sorted(unexpected)[:10]}"
    assert missing == exp_missing, (
        f"missing keys != expected new heads.\n  extra missing: {sorted(missing - exp_missing)[:20]}\n"
        f"  expected but present: {sorted(exp_missing - missing)[:20]}")
    assert dropped == exp_dropped, (
        f"dropped source keys != expected old heads.\n  extra dropped: {sorted(dropped - exp_dropped)[:20]}\n"
        f"  expected but loaded: {sorted(exp_dropped - dropped)[:20]}")
    assert len(remapped) + len(dropped) == len(sd) == 432, (len(remapped), len(dropped), len(sd))

    # movable arm: point output layers must be zero (semantics change) - enforce explicitly
    if arm == "movable-reference":
        with torch.no_grad():
            for head in model.decoder.point_head:
                nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)
    # new heads: wh output zero-init, class bias prior 0.01 (already set by init_weights, re-assert)
    with torch.no_grad():
        for head in model.decoder.wh_head:
            assert float(head[-1].weight.abs().sum()) == 0.0 and float(head[-1].bias.abs().sum()) == 0.0
        import math
        assert torch.allclose(model.decoder.class_head.bias, torch.full_like(model.decoder.class_head.bias, -math.log(99.0)))

    n_loaded_params = sum(v.numel() for v in remapped.values())
    n_model_params = sum(p.numel() for p in model.parameters())
    n_new_params = sum(model_sd[k].numel() for k in exp_missing)
    report = {
        "path": path, "arm": arm,
        "source_tensors": len(sd), "loaded_tensors": len(remapped), "dropped_tensors": len(dropped),
        "loaded_params": int(n_loaded_params), "model_params": int(n_model_params), "new_params": int(n_new_params),
        "coverage": n_loaded_params / n_model_params,
        "missing_keys": sorted(missing), "dropped_source_keys": sorted(dropped),
    }
    if verbose:
        print(f"[hf5class] arm={arm} loaded {report['loaded_tensors']}/{report['source_tensors']} tensors "
              f"({report['loaded_params']:,} params, coverage {report['coverage']*100:.2f}% of {report['model_params']:,}); "
              f"dropped {report['dropped_tensors']} source tensors; new params {report['new_params']:,}")
    return report
