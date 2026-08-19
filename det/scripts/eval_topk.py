"""Auxiliary evaluation with production top-k (default 4000) and matching COCO maxDets (§5.4).

The official A/B metric stays num_top_queries=2000 / maxDets [500,1000,2000] (trainer). This script runs the
same Dome ``evaluate`` on a checkpoint with a *separately named* output (never overwrites the official eval):
    <output-dir>/eval_top{K}.pth  and  <output-dir>/eval_top{K}.json (13 stats)
YAMLConfig only allows a fixed evaluator name list, so the maxDets override is done here on the evaluator instance.

Usage:
  python det/scripts/eval_topk.py -c det/configs/LSP-T-combined.yml -r <ckpt.pth> --output-dir <dir> [--topk 4000] [--split val|test]
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lsp_det  # noqa: F401
import torch
from src.core import YAMLConfig, yaml_utils
from src.misc import dist_utils
from src.solver import TASKS
from src.solver.det_engine import evaluate

p = argparse.ArgumentParser()
p.add_argument("-c", "--config", required=True)
p.add_argument("-r", "--resume", required=True, help="checkpoint (.pth with 'ema'/'model')")
p.add_argument("--output-dir", required=True)
p.add_argument("--topk", type=int, default=4000)
p.add_argument("--split", default="val", choices=["val", "test"])
p.add_argument("--tf32", default="keep", choices=["keep", "off"])
p.add_argument("-u", "--update", nargs="+", default=[])
a = p.parse_args()
if a.tf32 == "off":
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.set_float32_matmul_precision("highest")

dist_utils.setup_distributed(0, "builtin", seed=0)
upd = yaml_utils.parse_cli([f"DomePostProcessor.num_top_queries={a.topk}"] + a.update)
if a.split == "test":
    upd = yaml_utils.merge_dict(upd, {"val_dataloader": {"dataset": {"ann_file": "/home/work/.mnt/combined_all_v1_bundle/test_coco.json"}}})
upd.update({"resume": a.resume, "output_dir": a.output_dir, "use_amp": False, "test_only": True})
cfg = YAMLConfig(a.config, **upd)
solver = TASKS[cfg.yaml_cfg["task"]](cfg)
solver.eval()  # builds model/EMA/loader/evaluator and loads the checkpoint

# maxDets [K/4, K/2, K] so stats[0..] are reported at K
K = a.topk
ev = solver.evaluator
for ce in ev.coco_eval.values():
    ce.params.maxDets = [K // 4, K // 2, K]
_orig_cleanup = ev.cleanup
def _cleanup():
    _orig_cleanup()
    for ce in ev.coco_eval.values():
        ce.params.maxDets = [K // 4, K // 2, K]
ev.cleanup = _cleanup

module = solver.ema.module if solver.ema else solver.model
stats, coco_evaluator = evaluate(module, solver.criterion, solver.postprocessor, solver.val_dataloader, ev, solver.device, epoch=0, output_dir=None)
os.makedirs(a.output_dir, exist_ok=True)
tag = f"{a.split}_top{K}"
if dist_utils.is_main_process():
    torch.save(coco_evaluator.coco_eval["bbox"].eval, os.path.join(a.output_dir, f"eval_{tag}.pth"))
    with open(os.path.join(a.output_dir, f"eval_{tag}.json"), "w") as f:
        json.dump({"topk": K, "maxDets": [K // 4, K // 2, K], "split": a.split, "checkpoint": a.resume, "coco_eval_bbox": stats["coco_eval_bbox"]}, f, indent=2)
    print(f"[eval_topk] {tag}: AP={stats['coco_eval_bbox'][0]:.4f} AP50={stats['coco_eval_bbox'][1]:.4f} AR@{K}={stats['coco_eval_bbox'][8]:.4f} -> {a.output_dir}/eval_{tag}.json")
dist_utils.cleanup()
