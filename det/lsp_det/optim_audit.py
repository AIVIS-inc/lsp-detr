"""Optimizer param-group audit (§7): print every group (names/count/lr/wd) and assert

  * every trainable parameter is assigned to exactly one group (regexes mutually exclusive),
  * no ``backbone`` parameter sits in a base-LR group,
  * backbone norms get lr 1.25e-5 / wd 0, non-backbone norms get wd 0 (bias keeps default wd),
  * frozen tensors (requires_grad=False) are exactly the configured freeze set.

The grouping logic mirrors ``YAMLConfig.get_optim_params`` (first-match wins is *not* used there:
a param matching two regexes would be added to both groups) - hence the exclusivity assert.
"""

from __future__ import annotations

import re
from typing import Dict, List

import torch.nn as nn

__all__ = ["audit_param_groups"]


def audit_param_groups(model: nn.Module, optimizer_cfg: dict, expect: dict | None = None, verbose: bool = True,
                       print_names: bool = False) -> List[Dict]:
    base_lr = float(optimizer_cfg["lr"])
    base_wd = float(optimizer_cfg.get("weight_decay", 0.0))
    named = [(k, v) for k, v in model.named_parameters()]
    trainable = [(k, v) for k, v in named if v.requires_grad]
    frozen = [k for k, v in named if not v.requires_grad]

    groups = []
    assigned: Dict[str, List[int]] = {k: [] for k, _ in trainable}
    for gi, pg in enumerate(optimizer_cfg["params"]):
        pat = pg["params"]
        names = [k for k, _ in trainable if len(re.findall(pat, k)) > 0]
        for k in names:
            assigned[k].append(gi)
        groups.append({"idx": gi, "pattern": pat, "lr": float(pg.get("lr", base_lr)),
                       "weight_decay": float(pg.get("weight_decay", base_wd)), "names": names,
                       "n_tensors": len(names), "n_params": sum(dict(trainable)[k].numel() for k in names)})
    rest = [k for k, g in assigned.items() if len(g) == 0]
    groups.append({"idx": len(groups), "pattern": "<default/unmatched>", "lr": base_lr, "weight_decay": base_wd,
                   "names": rest, "n_tensors": len(rest), "n_params": sum(dict(trainable)[k].numel() for k in rest)})

    dup = {k: g for k, g in assigned.items() if len(g) > 1}
    if verbose:
        print("=" * 100)
        print(f"[optim audit] trainable tensors={len(trainable)} ({sum(v.numel() for _, v in trainable):,} params), "
              f"frozen tensors={len(frozen)} ({sum(dict(named)[k].numel() for k in frozen):,} params), base lr={base_lr} wd={base_wd}")
        for g in groups:
            print(f"  group[{g['idx']}] lr={g['lr']:.3e} wd={g['weight_decay']:.3e} tensors={g['n_tensors']:5d} params={g['n_params']:>12,}  pattern={g['pattern']}")
            if print_names:
                for k in g["names"]:
                    print(f"      {k}")
        if frozen and print_names:
            print("  frozen:")
            for k in frozen:
                print(f"      {k}")
        print("=" * 100)
    assert not dup, f"parameters matched by more than one optimizer group (regexes not exclusive): {list(dup.items())[:10]}"

    # policy checks
    for g in groups:
        for k in g["names"]:
            is_bb = "backbone" in k
            is_norm = "norm" in k
            if is_bb:
                assert g["lr"] < base_lr, f"backbone param {k} in base-LR group[{g['idx']}] ({g['pattern']})"
                if is_norm:
                    assert g["weight_decay"] == 0.0, f"backbone norm {k} has wd {g['weight_decay']}"
            else:
                assert g["lr"] == base_lr, f"non-backbone param {k} not at base lr (group[{g['idx']}])"
                if is_norm:
                    assert g["weight_decay"] == 0.0, f"non-backbone norm {k} has wd {g['weight_decay']}"
                elif not k.endswith("bias"):
                    assert g["weight_decay"] == base_wd, f"non-norm param {k} wd {g['weight_decay']} != base {base_wd}"
    if expect:
        if "backbone_lr" in expect:
            for g in groups:
                if any("backbone" in k for k in g["names"]):
                    assert abs(g["lr"] - expect["backbone_lr"]) < 1e-12, (g["lr"], expect["backbone_lr"])
        if "min_groups" in expect:
            assert sum(1 for g in groups if g["n_tensors"] > 0) >= expect["min_groups"], "fewer non-empty groups than expected"
        if "frozen_prefixes" in expect:
            for k in frozen:
                assert any(k.startswith(p) for p in expect["frozen_prefixes"]), f"unexpected frozen tensor {k}"
            for k, v in named:
                if any(k.startswith(p) for p in expect["frozen_prefixes"]):
                    assert not v.requires_grad, f"{k} should be frozen"
    return groups
