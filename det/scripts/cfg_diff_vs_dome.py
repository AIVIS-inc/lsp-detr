"""P2 gate: key-level diff of the resolved LSP cfg against the reference Dome HER2 run
(``AIVIS-Dome-DETR/train_run.log:32`` 'cfg:' line, i.e. the actual `YAMLConfig.__dict__` printed at launch).

Only differences in the ALLOWED set (model-specific blocks, ImageNet normalisation, dummy removal,
dataset paths / num_classes, output_dir, optimizer 3-group split) may remain; anything else fails.

Usage: python det/scripts/cfg_diff_vs_dome.py [--config det/configs/LSP-T-combined.yml] [--dome-log .../train_run.log]
Simulates the launcher CLI overrides (epoches=30, total_batch_size 8/8, seed 0, output-dir).
"""
import argparse, ast, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lsp_det  # noqa: F401
from src.core import YAMLConfig, yaml_utils

p = argparse.ArgumentParser()
p.add_argument("--config", default=os.path.join(lsp_det.DET_ROOT, "configs", "LSP-T-combined.yml"))
p.add_argument("--dome-log", default=None,
               help="train_run.log of the reference Dome HER2 run (its 'cfg:' line); lives in the AIVIS-DETECTION checkout, not in this repo")
p.add_argument("--arm", default="strict-local")
a = p.parse_args()
if not a.dome_log or not os.path.isfile(a.dome_log):
    sys.exit("pass --dome-log <AIVIS-Dome-DETR run>/train_run.log (the reference Dome run's log is not vendored in det/third_party/dome)")

# ---- reference cfg (Dome run) ---------------------------------------------------------------
with open(a.dome_log) as f:
    lines = f.readlines()
ref_line = next(l for l in lines if l.startswith("cfg:  {"))
ref = ast.literal_eval(ref_line[len("cfg:  "):].strip())

# ---- ours (same CLI overrides as the launcher) ------------------------------------------------
upd = yaml_utils.parse_cli(["epoches=30", "train_dataloader.total_batch_size=8", "val_dataloader.total_batch_size=8",
                            f"LSPDetrDetection.center_mode={a.arm}"])
# use_amp=False / test_only=False: argparse store_true defaults, forwarded exactly like Dome's train.py
upd.update({"config": a.config, "seed": 0, "test_only": False, "use_amp": False, "print_method": "builtin", "print_rank": 0,
            "output_dir": "output/[combined]LSP-T_strict-local_H100x8_hf5class-ft_30ep"})
cfg = YAMLConfig(a.config, **upd)
ours = cfg.__dict__


def flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(d, list) and d and all(isinstance(x, dict) for x in d):
        for i, v in enumerate(d):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = d
    return out


fr, fo = flatten(ref), flatten(ours)
keys = sorted(set(fr) | set(fo))

# allowed difference classes -> regex on the flattened key
ALLOWED = [
    ("model-specific block", r"^yaml_cfg\.(DOME|HGNetv2|HybridEncoder|DomeTransformer|DomeCriterion|LSPDetrDetection|LSPCriterion)\b"),
    ("model/criterion name", r"^yaml_cfg\.(model|criterion)$"),
    ("dataset paths / classes", r"^yaml_cfg\.(num_classes|train_dataloader\.dataset\.dataset\.(img_folder|ann_file)|val_dataloader\.dataset\.(img_folder|ann_file))$"),
    ("dummy removal (NoDummy crop + explicit wrapper crop_size)", r"^yaml_cfg\.train_dataloader\.dataset\.(crop_size|dataset\.transforms\.ops\[0\]\.type)$"),
    ("ImageNet normalize (extra last op)", r"^yaml_cfg\.(train_dataloader\.dataset\.dataset\.transforms\.ops\[8\]|val_dataloader\.dataset\.transforms\.ops\[2\])"),
    ("optimizer 3-group split", r"^yaml_cfg\.optimizer\.params\[[12]\]"),
    ("run identity", r"^(output_dir|yaml_cfg\.(output_dir|config|__include__(\[\d+\])?))$"),
]

rows, bad = [], []
for k in keys:
    if k in fr and k in fo and fr[k] == fo[k]:
        continue
    tag = next((name for name, pat in ALLOWED if re.search(pat, k)), None)
    rows.append((k, fr.get(k, "<absent>"), fo.get(k, "<absent>"), tag))
    if tag is None:
        bad.append(k)

print(f"reference: {a.dome_log}\nours     : {a.config} (arm={a.arm})")
print(f"flattened keys: ref={len(fr)} ours={len(fo)} identical={sum(1 for k in keys if k in fr and k in fo and fr[k]==fo[k])} differing={len(rows)}")
print("-" * 120)
for k, r, o, tag in rows:
    print(f"[{'OK ' if tag else 'BAD'}] {k}\n      dome: {r!r}\n      lsp : {o!r}\n      why : {tag}")
print("-" * 120)
same_important = ["yaml_cfg.use_ema", "yaml_cfg.ema.decay", "yaml_cfg.ema.warmups", "yaml_cfg.ema.start", "yaml_cfg.clip_max_norm",
                  "yaml_cfg.use_amp", "yaml_cfg.use_focal_loss", "yaml_cfg.epoches", "yaml_cfg.lr_scheduler.milestones",
                  "yaml_cfg.optimizer.lr", "yaml_cfg.optimizer.weight_decay", "yaml_cfg.optimizer.params[0].lr",
                  "yaml_cfg.DomePostProcessor.num_top_queries", "yaml_cfg.train_dataloader.num_workers", "yaml_cfg.train_dataloader.total_batch_size",
                  "yaml_cfg.checkpoint_freq", "yaml_cfg.sync_bn", "yaml_cfg.find_unused_parameters", "yaml_cfg.lr_warmup_scheduler.warmup_duration"]
for k in same_important:
    assert k in fo and k in fr and fo[k] == fr[k], (k, fr.get(k), fo.get(k))
print("recipe keys identical:", ", ".join(k.split("yaml_cfg.")[-1] for k in same_important))
if bad:
    print(f"\nFAIL: {len(bad)} unexplained differences: {bad}")
    sys.exit(1)
print("\nPASS: every difference is model-specific / normalisation / dummy-removal / dataset / run identity.")
