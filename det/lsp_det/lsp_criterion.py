"""``LSPCriterion``: DomeCriterion restricted to the LSP output contract (final + 5 aux) (§5.3).

Reuses ``loss_labels_vfl`` / ``loss_boxes`` / ``_get_go_indices`` / ``_get_match_pool`` from Dome and
keeps the recipe semantics: boxes (L1/GIoU) use the union match over all 6 layers with
``num_boxes_go`` normalisation (uni_set), VFL uses each layer's own match with ``num_boxes``.
No ``pre_outputs`` / ``enc_aux_outputs`` / ``dn_outputs`` / ``defe`` keys are read.
"""

from __future__ import annotations

import torch
import torch.distributed

from src.core import register
from src.misc.dist_utils import get_world_size, is_dist_available_and_initialized
from src.zoo.dome.dome_criterion import DomeCriterion

try:
    from src.zoo.dome.dome_criterion import _get_match_pool
except ImportError:  # $DOME_ROOT points at a Dome checkout without the thread-pool matcher patch
    def _get_match_pool():  # sequential matching: result-identical, slower (det/third_party/dome/VENDORED.md)
        return None

__all__ = ["LSPCriterion"]


@register()
class LSPCriterion(DomeCriterion):
    __share__ = ["num_classes"]
    __inject__ = ["matcher"]

    def __init__(self, matcher, weight_dict, losses, alpha=0.75, gamma=2.0, num_classes=2,
                 boxes_weight_format=None, share_matched_indices=False, use_uni_set=True):
        super().__init__(matcher=matcher, weight_dict=weight_dict, losses=losses, alpha=alpha, gamma=gamma,
                         num_classes=num_classes, boxes_weight_format=boxes_weight_format,
                         share_matched_indices=share_matched_indices, use_uni_set=use_uni_set)
        for l in self.losses:
            assert l in ("vfl", "boxes", "focal", "mal"), f"LSPCriterion does not support loss '{l}' (FGL/DDF are FDR-only)"

    # ---- §6.1 empty-crop / GT statistics (accumulated per epoch, printed at the last step of every epoch) ----
    def _stats_update(self, targets, device, **kwargs):
        if not self.training:
            return
        if not hasattr(self, "_ep_stats"):
            self._ep_stats = None
        if self._ep_stats is None:
            self._ep_stats = torch.zeros(4 + self.num_classes, dtype=torch.float64, device=device)  # crops, empty crops, all-empty batches, boxes, per-class
        n = [len(t["labels"]) for t in targets]
        s = self._ep_stats
        s[0] += len(targets); s[1] += sum(1 for x in n if x == 0); s[2] += float(all(x == 0 for x in n)); s[3] += sum(n)
        for t in targets:
            if len(t["labels"]):
                s[4:] += torch.bincount(t["labels"], minlength=self.num_classes).to(s.dtype)[: self.num_classes]
        step, epoch_step, epoch = kwargs.get("step"), kwargs.get("epoch_step"), kwargs.get("epoch")
        if step is not None and epoch_step is not None and step == epoch_step - 1:
            tot = s.clone()
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(tot)
            tot = tot.tolist()
            print(f"[LSPCriterion][epoch {epoch}] crops={int(tot[0])} empty_crops={int(tot[1])} ({100*tot[1]/max(tot[0],1):.2f}%) "
                  f"all_empty_batches={int(tot[2])} gt_boxes={int(tot[3])} per_class={[int(v) for v in tot[4:]]} "
                  f"(all ranks; batch=per-rank batch)")
            self._ep_stats = None

    def _num_boxes(self, n: int, device) -> float:
        t = torch.as_tensor([n], dtype=torch.float, device=device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(t)
        return torch.clamp(t / get_world_size(), min=1).item()

    def forward(self, outputs, targets, **kwargs):
        assert "pred_logits" in outputs and "pred_boxes" in outputs and "aux_outputs" in outputs, list(outputs.keys())
        device = outputs["pred_logits"].device
        outputs_without_aux = {k: v for k, v in outputs.items() if "aux" not in k}
        self._clear_cache()
        self._stats_update(targets, device, **kwargs)

        # match final + every aux layer (independent -> thread pool, order preserved)
        to_match = [outputs_without_aux] + list(outputs["aux_outputs"])
        pool = _get_match_pool()
        if pool is not None and len(to_match) > 1:
            def _match_one(o):
                dev = o["pred_logits"].device
                if dev.type == "cuda":
                    torch.cuda.set_device(dev)
                return self.matcher(o, targets)["indices"]
            matched = list(pool.map(_match_one, to_match))
        else:
            matched = [self.matcher(o, targets)["indices"] for o in to_match]
        indices, cached_indices = matched[0], matched[1:]
        indices_go = self._get_go_indices(indices, cached_indices)

        num_boxes_go = self._num_boxes(sum(len(x[0]) for x in indices_go), device)
        num_boxes = self._num_boxes(sum(len(t["labels"]) for t in targets), device)

        losses = {}
        for loss in self.losses:
            use_uni = self.use_uni_set and loss == "boxes"
            ind, nb = (indices_go, num_boxes_go) if use_uni else (indices, num_boxes)
            meta = self.get_loss_meta_info(loss, outputs, targets, ind)
            l_dict = self.get_loss(loss, outputs, targets, ind, nb, **meta)
            losses.update({k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict})

        for i, aux in enumerate(outputs["aux_outputs"]):
            for loss in self.losses:
                use_uni = self.use_uni_set and loss == "boxes"
                ind, nb = (indices_go, num_boxes_go) if use_uni else (cached_indices[i], num_boxes)
                meta = self.get_loss_meta_info(loss, aux, targets, ind)
                l_dict = self.get_loss(loss, aux, targets, ind, nb, **meta)
                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                losses.update({k + f"_aux_{i}": v for k, v in l_dict.items()})

        losses = {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}
        return losses
