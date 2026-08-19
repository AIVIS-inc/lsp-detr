"""P3-2: checkpoint parity at 256 input - ported trunk vs the HF snapshot (``hf-5class/modeling.py``).

The snapshot cannot import in this env (``transformers.utils.backbone_utils.load_backbone`` is gone), so it is
loaded as a package with a shim ``load_backbone`` that builds the same offline Swinv2 (image_size=256, i.e. the
hub config the snapshot was trained/served with). Both models load ``model.safetensors`` fully (the port with
``backbone_image_size=256`` so the Swin window/shift geometry matches at 256).

Compared (eval mode, same real 256 crop):
  * backbone feature maps (4 levels)      -> exact
  * decoder embeddings (final tgt)        -> |diff| max / rel
  * points (normalized final centres)     -> |diff| max (pixels at 256)
  * absolute_points (pixel query coords)
Then the intended cached-P deviation is demonstrated: perturb S in one layer of both models -> the snapshot's
eval output does not move (stale cached P), the port's does and equals its own train-mode (solve) output.
"""
import argparse, importlib.util, json, math, os, sys, types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lsp_det  # noqa: F401
import torch
from safetensors.torch import load_file

from lsp_det.lsp_detr_det import LSPDetrDetection, build_swinv2_backbone
from lsp_det.transforms import IMAGENET_MEAN, IMAGENET_STD

p = argparse.ArgumentParser()
p.add_argument("--device", default="cuda:0")
p.add_argument("--out", default=os.path.join(lsp_det.DET_ROOT, "logs", "p3_2_parity.json"))
p.add_argument("--tf32", default="off", choices=["off", "keep"], help="off: pure FP32 for the parity comparison (default); keep: machine default (TF32 override on)")
a = p.parse_args()
dev = torch.device(a.device)
torch.manual_seed(0)
tf32_default = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.get_float32_matmul_precision())
if a.tf32 == "off":
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.set_float32_matmul_precision("highest")
print(f"TF32 machine default (matmul, cudnn, precision) = {tf32_default}; parity run uses matmul tf32={torch.backends.cuda.matmul.allow_tf32}")

# ---------------------------------------------------------------- HF snapshot as a package with shim
import transformers.utils.backbone_utils as bu
def _shim_load_backbone(config):
    return build_swinv2_backbone(image_size=256, drop_path_rate=0.1)
bu.load_backbone = _shim_load_backbone
if not hasattr(bu, "verify_backbone_config_arguments"):
    bu.verify_backbone_config_arguments = lambda **kw: None

snap_dir = os.path.join(lsp_det.LSP_REPO_ROOT, "hf-5class")
pkg = types.ModuleType("hf5class_snapshot"); pkg.__path__ = [snap_dir]; sys.modules["hf5class_snapshot"] = pkg
def _load(name):
    spec = importlib.util.spec_from_file_location(f"hf5class_snapshot.{name}", os.path.join(snap_dir, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod); return mod
snap_cfg_mod = _load("configuration"); snap_mod = _load("modeling")
snap_cfg = snap_cfg_mod.LSPDetrConfig.from_pretrained(snap_dir)
snap = snap_mod.LSPDetrModel(snap_cfg)
sd = load_file(os.path.join(snap_dir, "model.safetensors"))
res = snap.load_state_dict(sd, strict=True)
snap = snap.to(dev).eval()
print(f"snapshot: {sum(p.numel() for p in snap.parameters()):,} params, strict load ok, qbs={snap.query_block_size}")

# ---------------------------------------------------------------- port at 256 geometry
port = LSPDetrDetection(num_classes=2, backbone_image_size=256, backbone_drop_path_rate=0.1, pretrained=lsp_det.HF5CLASS_CKPT,
                        wh_prior_px=[14.0, 14.0], center_mode="strict-local", input_norm_check="off").to(dev).eval()

# ---------------------------------------------------------------- input: real 256 crop from a val image
from src.data.dataset import CocoDetection
import PIL.Image
val_ann = "/home/work/.mnt/combined_all_v1_bundle/val_coco.json"
ds = CocoDetection(img_folder="/home/work/.mnt/combined_all_v1_bundle", ann_file=val_ann, transforms=None, return_masks=False)
img, _ = ds.load_item(0)
img = img.crop((640, 640, 896, 896))
x = torch.from_numpy(__import__("numpy").asarray(img)).permute(2, 0, 1).float().div(255)
x = ((x - torch.tensor(IMAGENET_MEAN)[:, None, None]) / torch.tensor(IMAGENET_STD)[:, None, None]).unsqueeze(0).to(dev)
print("input", tuple(x.shape), f"range [{float(x.min()):.2f},{float(x.max()):.2f}]")

report = {}
def cmp(name, A, B, scale=1.0):
    d = (A.float() - B.float()).abs()
    rel = float(d.max() / (B.float().abs().max() + 1e-12))
    report[name] = {"max_abs": float(d.max()) * scale, "mean_abs": float(d.mean()) * scale, "max_rel": rel, "shape": list(A.shape)}
    print(f"  {name:22s} shape={tuple(A.shape)} max|d|={float(d.max())*scale:.3e} mean|d|={float(d.mean())*scale:.3e} rel={rel:.2e}")

with torch.no_grad():
    # backbone
    fs = snap.backbone(x).feature_maps; fp = port.backbone(x).feature_maps
    print("backbone feature maps:")
    for i, (u, v) in enumerate(zip(fs, fp)):
        cmp(f"backbone.stage{i+1}", v, u)
    # full forward
    os_ = snap(x)
    op = port(x)
    print("decoder outputs (eval mode):")
    cmp("embeddings", op["embeddings"], os_["embeddings"])
    cmp("points(norm)", op["pred_center"], os_["points"])
    cmp("points(px@256)", op["pred_center"], os_["points"], scale=256.0)
    # snapshot absolute_points are the *last-layer detached* refs (px); port equivalent:
    # recompute from port's final centre: centre_norm*256
    Q = op["pred_logits"].shape[1]
    assert Q == 324 and os_["logits"].shape[1] == 324, (Q, os_["logits"].shape)
    # class logits cannot match (6-class -> 2-class); wh has no counterpart.

    # ---- intended deviation: stale cached P in the snapshot -------------------------------------
    print("cached-P deviation check (perturb S in layer 0 self-attention of both, small delta):")
    delta = torch.randn(32, 32, generator=torch.Generator().manual_seed(1)).to(dev) * 0.05
    snap_pe = snap.decode_head.layers[0].self_attention.pe
    port_pe = port.decoder.layers[0].self_attention.pe
    snap_pe.parametrizations.S.original.add_(delta)
    port_pe.parametrizations.S.original.add_(delta)
    # module level: snapshot cached P vs the P its current S implies; port P vs float64 truth
    I64 = torch.eye(32, dtype=torch.float64, device=dev)
    P_true = (I64 - snap_pe.S.double()) @ torch.linalg.inv(I64 + snap_pe.S.double())
    d_snap_P = float((snap_pe.P.double() - P_true).abs().max())          # cached_property -> stale
    d_port_P = float((port_pe.P().double() - P_true).abs().max())        # recomputed
    os2 = snap(x); op2 = port(x)
    d_snap = float((os2["embeddings"] - os_["embeddings"]).abs().max())
    d_port = float((op2["embeddings"] - op["embeddings"]).abs().max())
    # port eval vs port train (linalg.solve) path on the perturbed weights (drop_path off -> deterministic)
    port_train = LSPDetrDetection(num_classes=2, backbone_image_size=256, backbone_drop_path_rate=0.0, pretrained=lsp_det.HF5CLASS_CKPT,
                                  wh_prior_px=[14.0, 14.0], input_norm_check="off").to(dev)
    port_train.decoder.layers[0].self_attention.pe.parametrizations.S.original.add_(delta)
    port_train.train()
    op_train = port_train(x)
    d_port_vs_train = float((op2["embeddings"] - op_train["embeddings"]).abs().max())
    d_snap_vs_train = float((os2["embeddings"] - op_train["embeddings"]).abs().max())
    print(f"  |snapshot cached P - P(current S)| = {d_snap_P:.3e} (STALE)   |port P() - P64| = {d_port_P:.3e}")
    print(f"  snapshot eval output moved by {d_snap:.3e} (stale)   port eval output moved by {d_port:.3e}")
    print(f"  |port eval - port train(solve)| = {d_port_vs_train:.3e}   |snapshot eval - port train(solve)| = {d_snap_vs_train:.3e}")
    report["cachedP"] = {"snapshot_cachedP_vs_current": d_snap_P, "port_P_vs_f64": d_port_P, "snapshot_eval_delta": d_snap,
                         "port_eval_delta": d_port, "port_eval_vs_train": d_port_vs_train, "snapshot_eval_vs_train": d_snap_vs_train}

# ---- 1536-geometry backbone vs 256-geometry backbone at 256 input (expected to differ: window/shift) ------
with torch.no_grad():
    bb1536 = build_swinv2_backbone(image_size=1536).to(dev).eval()
    bb1536.load_state_dict({k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}, strict=True)
    f1536 = bb1536(x).feature_maps
    diffs = [float((u.float() - v.float()).abs().max()) for u, v in zip(fs, f1536)]
    print("backbone(image_size=1536) vs snapshot(256) at 256 input, per stage max|d|:", ["%.2e" % d for d in diffs],
          "(stage1/2 identical, stage3/4 differ by design: window16/shift8 vs window16-shift0 / window8)")
    report["geometry_1536_vs_256_at_256_input"] = diffs

ok = report["embeddings"]["max_abs"] < 1e-3 and report["points(px@256)"]["max_abs"] < 0.01 and all(report[f"backbone.stage{i}"]["max_abs"] < 1e-5 for i in range(1, 5))
ok = ok and report["cachedP"]["snapshot_eval_delta"] < 1e-6 and report["cachedP"]["snapshot_cachedP_vs_current"] > 1e-3 \
        and report["cachedP"]["port_P_vs_f64"] < 1e-5 and report["cachedP"]["port_eval_delta"] > 1e-6 and report["cachedP"]["port_eval_vs_train"] < 1e-3
report["tf32_mode"] = a.tf32
report["pass"] = bool(ok)
with open(a.out, "w") as f:
    json.dump(report, f, indent=2)
print("PASS" if ok else "FAIL", "->", a.out)
