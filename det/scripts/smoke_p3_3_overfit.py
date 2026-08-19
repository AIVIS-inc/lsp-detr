"""P3-3: 10-image overfit + strict-local vs movable-reference short ablation (§4.2, §8).

* 10 val images with 200 < GT < 1500 (val is 1536^2 with centre-672 GT; deterministic transforms:
  NoDummy crop (identity at 1536) -> /255 -> cxcywh normalize -> ImageNet normalize).
* Recipe optimizer/criterion/postprocessor from the real config (via YAMLConfig), grad clip 0.1, no EMA.
* Every --eval-every steps: AitodCocoEvaluator on the 10 images (official top-2000 + NMS), matched-IoU
  median (Hungarian on final outputs), greedy IoU>=0.5 recall overall / on "collision" GT (>=2 GT centres in
  the same qbs cell) / non-collision GT.
Gate: AP50 >= 0.95, matched IoU rising. Compare collision recall across arms.

Usage: python det/scripts/smoke_p3_3_overfit.py --arm strict-local --device cuda:0 --steps 600
"""
import argparse, json, math, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lsp_det  # noqa: F401
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.core import YAMLConfig, yaml_utils
from src.data.dataset import CocoDetection, AitodCocoEvaluator
from src.data.transforms.container import Compose
from src.data.dataloader import BatchImageCollateFunction
from src.solver.det_engine import evaluate
from src.zoo.dome.box_ops import box_cxcywh_to_xyxy, box_iou

p = argparse.ArgumentParser()
p.add_argument("--config", default=os.path.join(lsp_det.DET_ROOT, "configs", "LSP-T-combined.yml"))
p.add_argument("--arm", default="strict-local", choices=["strict-local", "movable-reference"])
p.add_argument("--device", default="cuda:0")
p.add_argument("--steps", type=int, default=600)
p.add_argument("--eval-every", type=int, default=100)
p.add_argument("--n-images", type=int, default=10)
p.add_argument("--gt-min", type=int, default=200)
p.add_argument("--gt-max", type=int, default=1500)
p.add_argument("--tf32", default="keep", choices=["keep", "off"])
p.add_argument("--out", default=None)
p.add_argument("-u", "--update", nargs="+", default=[])
a = p.parse_args()
out_path = a.out or os.path.join(lsp_det.DET_ROOT, "logs", f"p3_3_overfit_{a.arm}.json")
if a.tf32 == "off":
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.set_float32_matmul_precision("highest")
torch.manual_seed(0); np.random.seed(0)
dev = torch.device(a.device)

cfg = YAMLConfig(a.config, **yaml_utils.parse_cli([f"LSPDetrDetection.center_mode={a.arm}"] + a.update))
QBS = cfg.yaml_cfg["LSPDetrDetection"]["query_block_size"]

# ---------------------------------------------------------------- data (deterministic)
train_ops = [{"type": "RandomCropWithGridNoDummy", "crop_size": 1536, "center_gt_size": 672},
             {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
             {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
             {"type": "ImageNetNormalize"}]
val_ops = cfg.yaml_cfg["val_dataloader"]["dataset"]["transforms"]["ops"]
val_cfg = cfg.yaml_cfg["val_dataloader"]["dataset"]
base = CocoDetection(img_folder=val_cfg["img_folder"], ann_file=val_cfg["ann_file"], transforms=None, return_masks=False)
counts = [(len(base.coco.getAnnIds(imgIds=i)), idx) for idx, i in enumerate(base.ids)]
cands = [idx for n, idx in counts if a.gt_min < n < a.gt_max]
rng = np.random.RandomState(0)
sel = sorted(rng.choice(cands, size=a.n_images, replace=False).tolist())
print(f"selected {len(sel)} images (GT in ({a.gt_min},{a.gt_max})): idx={sel} gt={[counts[i][0] for i in sel]}")


class SubsetCoco(Dataset):
    """Subset of CocoDetection exposing .coco/.ids so det_engine.evaluate and the evaluator work unchanged."""
    def __init__(self, base, indices, transforms):
        self.base, self.indices, self._transforms = base, list(indices), transforms
        self.coco = base.coco
        self.ids = [base.ids[i] for i in self.indices]
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        img, tgt = self.base.load_item(self.indices[i])
        img, tgt, _ = self._transforms(img, tgt, self.base)
        return img, tgt
    def set_epoch(self, e): pass


train_ds = SubsetCoco(base, sel, Compose(ops=[dict(o) for o in train_ops]))
val_ds = SubsetCoco(base, sel, Compose(ops=[dict(o) for o in val_ops]))
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=BatchImageCollateFunction())
train_items = [train_ds[i] for i in range(len(train_ds))]  # cache (deterministic)
gts_norm = [t["boxes"].clone() for _, t in train_items]     # cxcywh normalized
gts_px = [box_cxcywh_to_xyxy(b) * 1536.0 for b in gts_norm]

# collision cells (>=2 GT centres in one qbs cell)
coll_masks = []
for b in gts_norm:
    cx, cy = b[:, 0] * 1536.0, b[:, 1] * 1536.0
    cell = (torch.floor(cx / QBS) + 1000 * torch.floor(cy / QBS)).long()
    _, inv, cnt = torch.unique(cell, return_inverse=True, return_counts=True)
    coll_masks.append(cnt[inv] >= 2)
n_gt = sum(len(b) for b in gts_norm); n_coll = int(sum(int(m.sum()) for m in coll_masks))
print(f"total GT {n_gt}, GT in collision cells {n_coll} ({100*n_coll/n_gt:.1f}%)")

# ---------------------------------------------------------------- model / crit / opt / eval
model = cfg.model.to(dev)
criterion = cfg.criterion.to(dev)
optimizer = cfg.optimizer
postprocessor = cfg.postprocessor.to(dev)
evaluator = AitodCocoEvaluator(coco_gt=base.coco, iou_types=["bbox"])
print(f"arm={a.arm} center_span={model.decoder.center_span} params={sum(p.numel() for p in model.parameters()):,}")


def greedy_recall(det_xyxy, det_scores, gt_xyxy, mask, thr=0.5):
    """greedy 1-1 matching by score; returns matched flags per GT."""
    matched = torch.zeros(len(gt_xyxy), dtype=torch.bool)
    if len(det_xyxy) == 0 or len(gt_xyxy) == 0:
        return matched
    iou, _ = box_iou(det_xyxy.cpu(), gt_xyxy.cpu())
    order = torch.argsort(det_scores.cpu(), descending=True)
    for d in order.tolist():
        row = iou[d].clone(); row[matched] = -1
        j = int(torch.argmax(row))
        if row[j] >= thr:
            matched[j] = True
    return matched


@torch.no_grad()
def full_eval(step):
    model.eval()
    stats, _ = evaluate(model, criterion, postprocessor, val_loader, evaluator, dev, epoch=0, output_dir=None)
    coco = stats["coco_eval_bbox"]
    # matched IoU (Hungarian on final outputs) + collision recall on postprocessed dets
    ious, rec_all, rec_coll, rec_ncoll = [], [], [], []
    for (img, tgt), gt_px, gt_n, cm in zip(train_items, gts_px, gts_norm, coll_masks):
        x = img.unsqueeze(0).to(dev)
        out = model(x)
        t = [{"labels": tgt["labels"].to(dev), "boxes": gt_n.to(dev)}]
        (si, ti), = criterion.matcher({"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}, t)["indices"]
        pb = box_cxcywh_to_xyxy(out["pred_boxes"][0][si.to(dev)]); gb = box_cxcywh_to_xyxy(gt_n.to(dev)[ti.to(dev)])
        iou = torch.diag(box_iou(pb, gb)[0]); ious.append(iou.cpu())
        res = postprocessor(out, torch.tensor([[1536, 1536]], device=dev))[0]
        m = greedy_recall(res["boxes"], res["scores"], gt_px, cm)
        rec_all.append(m); rec_coll.append(m[cm]); rec_ncoll.append(m[~cm])
    ious = torch.cat(ious)
    r = {"step": step, "AP": coco[0], "AP50": coco[1], "AR2000": coco[8],
         "matched_iou_median": float(ious.median()), "matched_iou_mean": float(ious.mean()),
         "recall_all": float(torch.cat(rec_all).float().mean()), "recall_collision": float(torch.cat(rec_coll).float().mean()),
         "recall_noncollision": float(torch.cat(rec_ncoll).float().mean())}
    print(f"[eval step {step}] AP={r['AP']:.4f} AP50={r['AP50']:.4f} AR@2000={r['AR2000']:.4f} matchedIoU med={r['matched_iou_median']:.3f} "
          f"recall all={r['recall_all']:.3f} collision={r['recall_collision']:.3f} non-coll={r['recall_noncollision']:.3f}")
    model.train()
    return r


history = {"arm": a.arm, "images": sel, "n_gt": n_gt, "n_collision_gt": n_coll, "evals": [], "loss": []}
history["evals"].append(full_eval(0))
model.train(); criterion.train()
t0 = time.time()
for step in range(1, a.steps + 1):
    img, tgt = train_items[(step - 1) % len(train_items)]
    x = img.unsqueeze(0).to(dev)
    targets = [{k: (v.to(dev) if hasattr(v, "to") else v) for k, v in tgt.items()}]
    out = model(x, targets=targets)
    loss_dict = criterion(out, targets, epoch=0, step=step)
    loss = sum(loss_dict.values())
    optimizer.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    optimizer.step()
    history["loss"].append(float(loss))
    if step % 20 == 0:
        print(f"step {step} loss {float(loss):.4f} (vfl {float(loss_dict['loss_vfl']):.3f} l1 {float(loss_dict['loss_bbox']):.3f} giou {float(loss_dict['loss_giou']):.3f}) {(time.time()-t0)/step:.2f}s/step")
    if step % a.eval_every == 0:
        history["evals"].append(full_eval(step))
        with open(out_path, "w") as f:
            json.dump(history, f, indent=2)

best = max(history["evals"], key=lambda r: r["AP50"])
ious = [r["matched_iou_median"] for r in history["evals"]]
history["best_AP50"] = best["AP50"]; history["best_step"] = best["step"]
history["matched_iou_rising"] = bool(ious[-1] > ious[0])
history["pass"] = bool(best["AP50"] >= 0.95 and history["matched_iou_rising"])
with open(out_path, "w") as f:
    json.dump(history, f, indent=2)
print(f"best AP50={best['AP50']:.4f} at step {best['step']}; matched IoU {ious[0]:.3f} -> {ious[-1]:.3f}; "
      f"final recall all/collision/non-coll = {history['evals'][-1]['recall_all']:.3f}/{history['evals'][-1]['recall_collision']:.3f}/{history['evals'][-1]['recall_noncollision']:.3f}")
print("PASS" if history["pass"] else "FAIL", "->", out_path)
