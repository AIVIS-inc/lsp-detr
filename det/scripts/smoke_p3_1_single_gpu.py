"""P3-1: single-GPU forward/backward smoke at 1536 (§8).

Checks: output contract/shapes, finite loss/grad/boxes, positive wh, no unused parameters, empty-target
and all-empty-batch step, peak memory (< 80 GB), step-time breakdown (fwd / criterion+matcher / bwd / opt).

Uses the *real* config (transforms + criterion + optimizer regexes) with the val split (already 1536, GT in
centre-672) unless --split train (loads the 4.8 GB train json). Picks the densest images by GT count so the
matcher cost matrices are worst-case.
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lsp_det  # noqa: F401
import torch
from src.core import YAMLConfig, yaml_utils, create, GLOBAL_CONFIG
from src.core.yaml_utils import merge_config
from src.data.transforms.container import Compose
from src.data.dataset import CocoDetection

p = argparse.ArgumentParser()
p.add_argument("--config", default=os.path.join(lsp_det.DET_ROOT, "configs", "LSP-T-combined.yml"))
p.add_argument("--split", default="val", choices=["val", "train"])
p.add_argument("--steps", type=int, default=6)
p.add_argument("--device", default="cuda:0")
p.add_argument("--dense-k", type=int, default=4, help="use the k densest images (by GT count)")
p.add_argument("--out", default=os.path.join(lsp_det.DET_ROOT, "logs", "p3_1_single_gpu.json"))
p.add_argument("-u", "--update", nargs="+", default=[])
a = p.parse_args()

torch.manual_seed(0)
dev = torch.device(a.device)
cfg = YAMLConfig(a.config, **yaml_utils.parse_cli(a.update))
gcfg = cfg.global_cfg

# ---- data: real train transform ops on the chosen split -----------------------------------------
ops = cfg.yaml_cfg["train_dataloader"]["dataset"]["dataset"]["transforms"]["ops"]
transforms = Compose(ops=[dict(o) for o in ops])
ds_cfg = cfg.yaml_cfg["train_dataloader"]["dataset"]["dataset"] if a.split == "train" else cfg.yaml_cfg["val_dataloader"]["dataset"]
t0 = time.time()
ds = CocoDetection(img_folder=ds_cfg["img_folder"], ann_file=ds_cfg["ann_file"], transforms=transforms, return_masks=False)
print(f"dataset {a.split}: {len(ds)} images, loaded in {time.time()-t0:.1f}s")
counts = sorted(((len(ds.coco.getAnnIds(imgIds=i)), idx) for idx, i in enumerate(ds.ids)), reverse=True)
dense_idx = [idx for _, idx in counts[: a.dense_k]]
print("densest images (GT count, idx):", counts[: a.dense_k])

# ---- model / criterion / optimizer from cfg ---------------------------------------------------------
model = cfg.model.to(dev)
criterion = cfg.criterion.to(dev)
optimizer = cfg.optimizer  # uses cfg.model params (same objects)
postprocessor = cfg.postprocessor.to(dev)
print(f"model params: {sum(p.numel() for p in model.parameters()):,} trainable {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
results = {"steps": [], "split": a.split, "config": a.config}


def one_step(samples, targets, tag):
    model.train(); criterion.train()
    torch.cuda.reset_peak_memory_stats(dev)
    torch.cuda.synchronize(dev)
    t_start = time.time()
    outputs = model(samples, targets=targets)
    torch.cuda.synchronize(dev); t_fwd = time.time()
    loss_dict = criterion(outputs, targets, epoch=0, step=0)
    torch.cuda.synchronize(dev); t_crit = time.time()
    loss = sum(loss_dict.values())
    optimizer.zero_grad()
    loss.backward()
    torch.cuda.synchronize(dev); t_bwd = time.time()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    optimizer.step()
    torch.cuda.synchronize(dev); t_opt = time.time()

    # contract checks
    B, Q = outputs["pred_logits"].shape[:2]
    assert outputs["pred_logits"].shape == (B, Q, 2) and outputs["pred_boxes"].shape == (B, Q, 4), (outputs["pred_logits"].shape, outputs["pred_boxes"].shape)
    assert Q == 108 * 108, Q
    assert len(outputs["aux_outputs"]) == 5 and all(o["pred_logits"].shape == (B, Q, 2) and o["pred_boxes"].shape == (B, Q, 4) for o in outputs["aux_outputs"])
    assert torch.isfinite(outputs["pred_boxes"]).all() and torch.isfinite(outputs["pred_logits"]).all()
    assert (outputs["pred_boxes"][..., 2:] > 0).all(), "non-positive wh"
    assert all(torch.isfinite(v) for v in loss_dict.values()), loss_dict
    unused = [n for n, p in trainable if p.grad is None]
    nonfinite = [n for n, p in trainable if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not nonfinite, nonfinite[:5]
    peak = torch.cuda.max_memory_allocated(dev) / 2**30
    rec = {"tag": tag, "num_gt": [int(len(t["labels"])) for t in targets], "loss": float(loss),
           "loss_vfl": float(loss_dict["loss_vfl"]), "loss_bbox": float(loss_dict["loss_bbox"]), "loss_giou": float(loss_dict["loss_giou"]),
           "grad_norm": float(gnorm), "t_fwd": t_fwd - t_start, "t_criterion": t_crit - t_fwd, "t_bwd": t_bwd - t_crit, "t_opt": t_opt - t_bwd,
           "t_step": t_opt - t_start, "peak_gb": peak, "unused_params": unused,
           "wh_px_median": float((outputs["pred_boxes"][..., 2] * samples.shape[-1]).median())}
    print(f"[{tag}] gt={rec['num_gt']} loss={rec['loss']:.4f} (vfl {rec['loss_vfl']:.3f} l1 {rec['loss_bbox']:.3f} giou {rec['loss_giou']:.3f}) "
          f"gnorm={rec['grad_norm']:.3f} fwd={rec['t_fwd']:.2f}s crit={rec['t_criterion']:.2f}s bwd={rec['t_bwd']:.2f}s opt={rec['t_opt']:.2f}s "
          f"step={rec['t_step']:.2f}s peak={peak:.1f}GB unused={len(unused)} wh_med={rec['wh_px_median']:.1f}px")
    results["steps"].append(rec)
    return outputs


def batch_from(indices):
    items = [ds[i] for i in indices]
    samples = torch.stack([it[0] for it in items]).to(dev)
    targets = [{k: (v.to(dev) if hasattr(v, "to") else v) for k, v in it[1].items()} for it in items]
    return samples, targets


# 1) compile warm-up + dense steps
for s in range(a.steps):
    samples, targets = batch_from([dense_idx[s % len(dense_idx)]])
    outputs = one_step(samples, targets, f"dense_step{s}")
    if s == 0:
        results["compile_warmup_step_s"] = results["steps"][-1]["t_step"]

# 2) empty target (all-empty batch)
samples, targets = batch_from([dense_idx[0]])
targets = [{**t, "labels": t["labels"][:0], "boxes": t["boxes"][:0]} for t in targets]
one_step(samples, targets, "all_empty_batch")
# 3) mixed batch of 2 (one empty) - checks B>1 path as well
samples2, targets2 = batch_from(dense_idx[:2])
targets2[1] = {**targets2[1], "labels": targets2[1]["labels"][:0], "boxes": targets2[1]["boxes"][:0]}
one_step(samples2, targets2, "b2_one_empty")

# 4) eval path + postprocessor (top-2000 + NMS) + all_class_scores contract
model.eval()
with torch.no_grad():
    samples, targets = batch_from([dense_idx[0]])
    out = model(samples, targets=None)
    res = postprocessor(out, torch.stack([t["orig_size"] for t in targets]))
    r = res[0]
    assert r["boxes"].shape[1] == 4 and r["all_class_scores"].shape[1] == 2 and len(r["labels"]) == len(r["scores"]) == len(r["boxes"])
    assert len(r["boxes"]) <= 2000
    print(f"[eval] postprocessor: {len(r['boxes'])} dets after top-2000 + NMS(0.7)/score 0.01; labels set {sorted(set(r['labels'].tolist()))}")
    results["eval_num_dets"] = int(len(r["boxes"]))
    # unique query duplication check: query-class flatten top-k may pick same query twice -> class-agnostic NMS removes exact duplicates
    results["eval_all_class_scores_shape"] = list(r["all_class_scores"].shape)

results["peak_gb_max"] = max(s["peak_gb"] for s in results["steps"])
results["step_time_after_warmup"] = [s["t_step"] for s in results["steps"][1:a.steps]]
results["unused_params_any"] = sorted({n for s in results["steps"] for n in s["unused_params"]})
results["pass"] = results["peak_gb_max"] < 80 and not results["unused_params_any"]
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w") as f:
    json.dump(results, f, indent=2)
print(f"peak GB max={results['peak_gb_max']:.1f}  step times (post-warmup)={['%.2f' % t for t in results['step_time_after_warmup']]}  unused={results['unused_params_any']}")
print("PASS" if results["pass"] else "FAIL", "->", a.out)
