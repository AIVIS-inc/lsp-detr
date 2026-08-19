"""LSP-DETR (bbox port) inference: images / directory / COCO json  ->  COCO-style predictions.

Counterpart of the Dome-DETR inference tools (tools/vis_inference.py, integrated_wsi_infer.py) for the LSP arm
trained by ``det/train.py``. Everything the *training/eval path* does is reproduced 1:1:

  preprocess   RGB -> Resize(eval size, whole-image mode) -> /255 -> ImageNet mean/std      (== val transforms)
  model        LSPDetrDetection built from the same yml (hf-5class init disabled) + checkpoint (EMA weights)
  postprocess  DomePostProcessor from the yml (top-2000, class-agnostic NMS 0.7, score>0.01) -> --conf-threshold

Input modes
  * single image, directory (recursive), *.txt list of paths, or COCO json (``images[].file_name`` + ``--img-folder``)
  * The recipe only has GT in the centre 672x672 of a 1536 crop, so the network only fires there. Hence
    ``--tiling auto`` (default, full coverage): every image is processed at native scale as a centred grid of
    ``--patch-size`` tiles (stride ``--step-size``, centre ``--filter-size`` ownership region, symmetric reflect padding
    exactly like the training small-image path); an image <= 672 px is a single centred tile.
    ``--tiling off`` (eval parity with the trainer): ONE forward per image - 1536 inputs as-is, smaller ones centred with
    reflect padding at native scale, larger ones resized to the model input (val ``Resize`` semantics; centre-only coverage).
    ``--tiling on`` == auto (kept for the Dome run_*.sh vocabulary).
Outputs (in --output-dir)
  * predictions{suffix}.json   {"meta", "categories", "images", "annotations": [{image_id, category_id, bbox xywh, score, ...}]}
  * per_image_counts{suffix}.csv   per-image detection counts by class (>= conf threshold)
  * vis{suffix}/...              overlays (--visualize)
  * coco_eval{suffix}.json       AitodCocoEvaluator stats (same evaluator/maxDets as the trainer) when --gt-json is
                                 given (evaluated on the *raw* postprocessor output, i.e. threshold independent)

Usage:  see det/run_inference.sh
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

DET_ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_CLASS_NAMES = ["Non-tumor", "Tumor"]  # combined_all_v1_bundle categories (id 0 / 1)
CLASS_COLORS = {0: (0, 90, 255), 1: (255, 0, 0), 2: (255, 200, 0), 3: (0, 200, 0), 4: (160, 160, 160)}


def _bool(x: str) -> bool:
    return str(x).strip().lower() in ("1", "true", "t", "yes", "y", "on")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # model
    p.add_argument("-c", "--config", default=os.path.join(DET_ROOT, "configs", "LSP-T-combined.yml"))
    p.add_argument("-r", "--resume", required=True, help="checkpoint .pth (best_stg1.pth / checkpointXXXX.pth / last.pth)")
    p.add_argument("--weights", default="ema", choices=["ema", "model"],
                   help="which weights to take from the checkpoint (ema = what the trainer evaluates; falls back to model)")
    p.add_argument("-u", "--update", nargs="+", default=[], help="extra yaml overrides, e.g. DomePostProcessor.num_top_queries=4000")
    # input
    p.add_argument("-i", "--input", required=True, help="image file | directory | *.txt list | COCO json")
    p.add_argument("--img-folder", default=None, help="root for relative paths in a COCO json / txt list (default: json's dir)")
    p.add_argument("--no-recursive", action="store_true", help="directory mode: do not descend into sub-directories")
    p.add_argument("--limit", type=int, default=None, help="only the first N images (smoke tests)")
    # output
    p.add_argument("-o", "--output-dir", required=True)
    p.add_argument("--output-suffix", default="", help="appended to every output file name, e.g. _ep3")
    p.add_argument("--visualize", type=_bool, default=False, help="write overlays to <output-dir>/vis<suffix>/")
    p.add_argument("--vis-limit", type=int, default=None, help="overlay at most N images")
    p.add_argument("--vis-score-threshold", type=float, default=None, help="default: --conf-threshold (cannot go below it)")
    p.add_argument("--no-class-scores", action="store_true", help="omit per-class score vectors from the json")
    p.add_argument("--class-names", default=None, help="comma separated, default from --gt-json categories or Non-tumor,Tumor")
    # detection thresholds (Dome run_*.sh vocabulary)
    p.add_argument("--conf-threshold", type=float, default=0.5, help="score threshold for saved/visualised/counted boxes")
    p.add_argument("--nms-iou-threshold", type=float, default=None, help="override DomePostProcessor.nms_iou_threshold (yml: 0.7)")
    p.add_argument("--nms-score-threshold", type=float, default=None, help="override DomePostProcessor.nms_score_threshold (yml: 0.01)")
    p.add_argument("--num-top-queries", type=int, default=None, help="override DomePostProcessor.num_top_queries (yml: 2000)")
    p.add_argument("--use-nms", type=_bool, default=None, help="override DomePostProcessor.use_nms (yml: true)")
    p.add_argument("--class-agnostic-nms", type=_bool, default=None, help="override DomePostProcessor.class_agnostic (yml: true)")
    p.add_argument("--include-non-tumor", type=_bool, default=True, help="False -> drop class 0 (Non-tumor) from the outputs")
    # geometry
    p.add_argument("--input-size", type=int, default=None, help="whole-image mode square input (default: yml eval_spatial_size, 1536)")
    p.add_argument("--tiling", default="auto", choices=["auto", "on", "off"],
                   help="auto/on: centred tile grid at native scale (full coverage); off: one forward per image (eval parity, centre-only)")
    p.add_argument("--allow-arm-mismatch", action="store_true",
                   help="only warn (instead of raising) when init_report.json next to the checkpoint disagrees with the built model")
    p.add_argument("--patch-size", type=int, default=1536, help="tile size (model input) in tiling mode")
    p.add_argument("--step-size", type=int, default=672, help="tile stride in tiling mode")
    p.add_argument("--filter-size", type=int, default=672, help="centre ownership region of a tile (box centre must fall inside)")
    p.add_argument("--tile-merge-nms", type=float, default=None,
                   help="optional class-agnostic NMS IoU applied over the merged tiles of one image (default: none, like the WSI pipeline)")
    # runtime
    p.add_argument("--gpu", default=None, help="sets CUDA_VISIBLE_DEVICES (Dome run_*.sh convention)")
    p.add_argument("-d", "--device", default="cuda")
    p.add_argument("-b", "--batch-size", type=int, default=4, help="images (whole mode) or tiles (tiling mode) per forward")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", default="none", choices=["none", "bf16", "fp16"], help="autocast dtype (default none == training/eval path)")
    p.add_argument("--tf32", default="keep", choices=["keep", "off", "on"], help="TF32 policy, same semantics as det/train.py")
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    # evaluation
    p.add_argument("--gt-json", default=None, help="COCO GT json -> AitodCocoEvaluator on the raw postprocessor output")
    return p.parse_args(argv)


# --------------------------------------------------------------------------------------- inputs
def _list_images_in_dir(root: str, recursive: bool) -> List[str]:
    out = []
    if recursive:
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if fn.lower().endswith(IMAGE_EXTS):
                    out.append(os.path.relpath(os.path.join(dp, fn), root))
    else:
        out = [fn for fn in os.listdir(root) if fn.lower().endswith(IMAGE_EXTS) and os.path.isfile(os.path.join(root, fn))]
    return sorted(out)


def collect_inputs(args) -> Tuple[List[dict], Optional[str], Optional[str]]:
    """Return (items, img_root, coco_json_used_as_listing). Each item: {file_name (relative), path, image_id?}."""
    inp = args.input
    if not os.path.exists(inp):
        raise FileNotFoundError(inp)
    items: List[dict] = []
    listing_json = None
    if os.path.isdir(inp):
        root = os.path.abspath(inp)
        for fn in _list_images_in_dir(root, recursive=not args.no_recursive):
            items.append({"file_name": fn, "path": os.path.join(root, fn)})
    elif inp.lower().endswith(".json"):
        listing_json = inp
        with open(inp) as f:
            d = json.load(f)
        root = os.path.abspath(args.img_folder or os.path.dirname(os.path.abspath(inp)))
        for im in d["images"]:
            items.append({"file_name": im["file_name"], "path": os.path.join(root, im["file_name"]),
                          "image_id": int(im["id"]), "width": im.get("width"), "height": im.get("height")})
    elif inp.lower().endswith(".txt"):
        root = os.path.abspath(args.img_folder) if args.img_folder else None
        with open(inp) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                path = line if (os.path.isabs(line) or root is None) else os.path.join(root, line)
                fn = os.path.relpath(path, root) if root and not os.path.isabs(line) else os.path.basename(path)
                items.append({"file_name": fn, "path": os.path.abspath(path)})
        root = root or os.getcwd()
    elif inp.lower().endswith(IMAGE_EXTS):
        root = os.path.dirname(os.path.abspath(inp))
        items.append({"file_name": os.path.basename(inp), "path": os.path.abspath(inp)})
    else:
        raise ValueError(f"unsupported --input {inp!r} (image / dir / .txt / .json)")
    seen, uniq = set(), []
    for it in items:
        key = os.path.realpath(it["path"])
        if key in seen:
            continue
        seen.add(key); uniq.append(it)
    if len(uniq) != len(items):
        print(f"[warn] {len(items) - len(uniq)} duplicate input paths dropped")
    items = uniq
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        raise RuntimeError(f"no images found in {inp!r}")
    return items, root, listing_json


def attach_gt_ids(items: List[dict], gt_json: str, img_root: str) -> Tuple["object", List[str], Dict[int, str]]:
    """Map every item to a GT image id (by relative file_name, then by basename). Returns (COCO, class_names, id->name)."""
    from pycocotools.coco import COCO
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(gt_json)
    by_name = {im["file_name"]: im["id"] for im in coco.dataset["images"]}
    by_base = {}
    for im in coco.dataset["images"]:
        by_base.setdefault(os.path.basename(im["file_name"]), []).append(im["id"])
    missing = []
    for it in items:
        carried = it.get("image_id")
        if carried is not None and carried in coco.imgs and \
                os.path.basename(coco.imgs[carried]["file_name"]) == os.path.basename(it["file_name"]):
            continue  # listing json == GT json (or consistent ids)
        iid = by_name.get(it["file_name"])
        if iid is None:
            cands = by_base.get(os.path.basename(it["file_name"]), [])
            iid = cands[0] if len(cands) == 1 else None
        if iid is None:
            missing.append(it["file_name"])
        else:
            it["image_id"] = int(iid)
    if missing:
        raise RuntimeError(f"{len(missing)} input images not found in --gt-json (first: {missing[:3]})")
    ids = [it["image_id"] for it in items]
    if len(set(ids)) != len(ids):
        raise RuntimeError("two input images map to the same --gt-json image id (COCO eval would be corrupted)")
    cats = {c["id"]: c["name"] for c in coco.dataset.get("categories", [])}
    names = [cats[k] for k in sorted(cats)] if cats else []
    return coco, names, cats


# --------------------------------------------------------------------------------------- data
class InferenceDataset:
    """Decodes (and, in the resize path, resizes) on CPU workers; padding/normalisation happen on the GPU (Runner).

    mode per image:  'tile'   tiling auto/on: centred tile grid at native scale (Runner.run_tiled)
                     'single' tiling off & max(w,h) <= input: one centred reflect-padded tile (Runner.run_tiled(single=True))
                     'whole'  tiling off & image larger than input: Resize -> one forward (val transform semantics)
    """

    def __init__(self, items: List[dict], input_size: int, tiling: str, patch_size: int):
        self.items = items
        self.input_size = int(input_size)
        self.tiling = tiling
        self.patch_size = int(patch_size)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        import torch
        from PIL import Image
        import torchvision.transforms.v2.functional as TF
        it = self.items[idx]
        base = {"idx": idx, "file_name": it["file_name"], "path": it["path"], "image_id": it.get("image_id", idx + 1)}
        try:
            im = Image.open(it["path"]).convert("RGB")
        except Exception as e:  # unreadable file must not kill a long run
            return {**base, "error": f"{type(e).__name__}: {e}", "orig_w": 0, "orig_h": 0, "mode": "error", "img": None}
        w, h = im.size
        if self.tiling in ("auto", "on"):
            mode = "tile"
        elif max(w, h) <= self.input_size:
            mode = "single"
        else:
            mode = "whole"
            # == Dome val transform `Resize` (torchvision v2, bilinear + antialias on PIL)
            im = TF.resize(im, [self.input_size, self.input_size])
        x = TF.pil_to_tensor(im).float() / 255.0  # == ConvertPILImage(float32, scale=True)
        return {**base, "img": x, "orig_w": w, "orig_h": h, "mode": mode}


def _identity_collate(batch):
    return batch


# --------------------------------------------------------------------------------------- model
def read_init_report(resume: str):
    """init_report.json written by det/train.py in the run dir (checkpoint dir or its parent). Returns (dict|None, path|None)."""
    d = os.path.dirname(os.path.abspath(resume))
    for cand in (os.path.join(d, "init_report.json"), os.path.join(os.path.dirname(d), "init_report.json")):
        if os.path.isfile(cand):
            try:
                with open(cand) as f:
                    return json.load(f), cand
            except Exception as e:
                print(f"[warn] cannot read {cand}: {e}")
    return None, None


def build_model_and_postprocessor(args):
    """YAMLConfig from the training yml -> LSPDetrDetection (no hf-5class init) + DomePostProcessor(deploy)."""
    import torch
    import lsp_det  # noqa: F401  (bootstraps Dome onto sys.path, registers the LSP components)
    from src.core import YAMLConfig, yaml_utils

    upd = yaml_utils.parse_cli(list(args.update))
    model_upd = {"pretrained": None,          # weights come from --resume; do not touch hf-5class/model.safetensors
                 "input_norm_check": "warn"}   # this script normalises explicitly (a uniform mid-grey first batch could false-alarm)
    # The arm (center_mode / movable_span_cells / wh_prior_px) is a CLI override at training time and leaves no trace in
    # the checkpoint (same tensor shapes) -> take it from the run's init_report.json unless the user overrides it via -u.
    report, report_path = read_init_report(args.resume)
    user_model_upd = upd.get("LSPDetrDetection", {}) if isinstance(upd.get("LSPDetrDetection"), dict) else {}
    if report:
        if report.get("center_mode") and "center_mode" not in user_model_upd:
            model_upd["center_mode"] = report["center_mode"]
        if report.get("center_mode") == "movable-reference" and report.get("center_span_cells") and "movable_span_cells" not in user_model_upd:
            model_upd["movable_span_cells"] = float(report["center_span_cells"])
        if report.get("wh_prior_px") and "wh_prior_px" not in user_model_upd:
            model_upd["wh_prior_px"] = [float(v) for v in report["wh_prior_px"]]
        print(f"[arm] init_report.json: {report_path} -> center_mode={report.get('center_mode')} "
              f"center_span_cells={report.get('center_span_cells')} wh_prior_px={report.get('wh_prior_px')} model={report.get('model')}")
    else:
        print(f"[warn] no init_report.json next to {args.resume}: trusting the yml/-u arm settings "
              f"(center_mode / wh_prior_px) - make sure they match the checkpoint")
    upd = yaml_utils.merge_dict(upd, {"LSPDetrDetection": model_upd})
    pp = {}
    if args.nms_iou_threshold is not None: pp["nms_iou_threshold"] = float(args.nms_iou_threshold)
    if args.nms_score_threshold is not None: pp["nms_score_threshold"] = float(args.nms_score_threshold)
    if args.num_top_queries is not None: pp["num_top_queries"] = int(args.num_top_queries)
    if args.use_nms is not None: pp["use_nms"] = bool(args.use_nms)
    if args.class_agnostic_nms is not None: pp["class_agnostic"] = bool(args.class_agnostic_nms)
    if pp:
        upd = yaml_utils.merge_dict(upd, {"DomePostProcessor": pp})
    cfg = YAMLConfig(args.config, **upd)

    eval_size = cfg.yaml_cfg.get("eval_spatial_size", [1536, 1536])
    input_size = int(args.input_size or eval_size[0])
    if isinstance(eval_size, (list, tuple)) and len(eval_size) == 2 and eval_size[0] != eval_size[1]:
        print(f"[warn] eval_spatial_size {eval_size} is not square; using {input_size}x{input_size}")

    ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
    which = args.weights
    if which == "ema" and not ("ema" in ckpt and isinstance(ckpt["ema"], dict) and "module" in ckpt["ema"]):
        print("[warn] checkpoint has no 'ema' -> using 'model' weights")
        which = "model"
    if which == "ema":
        state = ckpt["ema"]["module"]
    elif "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt  # raw state dict
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model = cfg.model
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint/model mismatch: missing={missing[:5]}... unexpected={unexpected[:5]}...")
    n_params = sum(p.numel() for p in model.parameters())
    epoch = ckpt.get("last_epoch", None)
    # cross-check the *resolved* arm against the report (config-only quantities that strict load cannot verify)
    resolved = {"model": cfg.yaml_cfg.get("model"), "center_mode": getattr(model, "center_mode", None),
                "center_span_cells": float(getattr(model.decoder, "center_span", float("nan"))),
                "wh_prior_px": [round(float(v), 4) for v in torch.exp(model.decoder.log_wh_prior).tolist()]}
    if report:
        problems = []
        for k in ("model", "center_mode"):
            if report.get(k) is not None and report[k] != resolved[k]:
                problems.append(f"{k}: report={report[k]} built={resolved[k]}")
        if report.get("center_span_cells") is not None and abs(float(report["center_span_cells"]) - resolved["center_span_cells"]) > 1e-6:
            problems.append(f"center_span_cells: report={report['center_span_cells']} built={resolved['center_span_cells']}")
        if report.get("wh_prior_px") is not None and any(abs(float(a) - b) > 1e-3 for a, b in zip(report["wh_prior_px"], resolved["wh_prior_px"])):
            problems.append(f"wh_prior_px: report={report['wh_prior_px']} built={resolved['wh_prior_px']}")
        if problems:
            msg = "[arm] built model disagrees with the checkpoint's init_report.json: " + "; ".join(problems)
            if args.allow_arm_mismatch:
                print("[warn] " + msg)
            else:
                raise RuntimeError(msg + " (pass --allow-arm-mismatch to override)")
    print(f"[model] {cfg.yaml_cfg.get('model')} params={n_params:,} weights={which} last_epoch={epoch} "
          f"center_mode={getattr(model, 'center_mode', None)} ckpt={args.resume}")
    postprocessor = cfg.postprocessor.deploy()
    print(f"[postprocessor] {postprocessor}")
    model.eval()
    if hasattr(model, "deploy"):
        model.deploy()
    meta = {"config": os.path.abspath(args.config), "checkpoint": os.path.abspath(args.resume), "weights": which,
            "last_epoch": epoch, "model": cfg.yaml_cfg.get("model"), "num_classes": int(cfg.yaml_cfg.get("num_classes", 2)),
            "arm": {**resolved, "init_report": report_path},
            "input_size": input_size, "postprocessor": {"num_top_queries": postprocessor.num_top_queries,
                                                        "use_nms": postprocessor.use_nms,
                                                        "nms_iou_threshold": postprocessor.nms_iou_threshold,
                                                        "nms_score_threshold": postprocessor.nms_score_threshold,
                                                        "class_agnostic": postprocessor.class_agnostic}}
    return model, postprocessor, cfg, input_size, meta


def apply_tf32_policy(mode: str) -> None:
    import torch
    if mode == "off":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    elif mode == "on":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    print(f"[tf32] mode={mode} matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32} "
          f"float32_matmul_precision={torch.get_float32_matmul_precision()}")


# --------------------------------------------------------------------------------------- tiling geometry
def tile_origins(size: int, patch: int, step: int, filt: int, single: bool = False) -> List[int]:
    """Origins (in un-padded image coords, usually negative) of a *centred* tile grid along one axis.

    Tile i owns the centre region [o_i + m, o_i + m + filt) with m = (patch - filt) // 2; the union of the ownership
    regions ((n-1)*step + filt wide, n = ceil(size/step)) is centred on [0, size), so borders are treated symmetrically
    and an image <= filt (n = 1) sits exactly where the training small-image path puts it (symmetric padding to patch).
    single=True forces n = 1 (one centred patch, --tiling off)."""
    m = (patch - filt) // 2
    n = 1 if single else max(1, math.ceil(size / step))
    span = (n - 1) * step + filt
    offset = (span - size) // 2 if not single else (patch - size) // 2 - m
    return [i * step - m - offset for i in range(n)]


def reflect_pad(img, pad_l: int, pad_r: int, pad_t: int, pad_b: int):
    """Reflect-pad [1,3,H,W] by possibly large amounts (iterative growth like Dome's _handle_small_image); the original
    content ends up at (pad_t, pad_l). Falls back to replicate only for a 1-pixel axis (reflect impossible)."""
    import torch.nn.functional as F

    def _axis(x, a, b, last):
        while a > 0 or b > 0:
            size = x.shape[-1] if last else x.shape[-2]
            if size == 1:
                return F.pad(x, (a, b, 0, 0) if last else (0, 0, a, b), mode="replicate")
            ga, gb = min(a, size - 1), min(b, size - 1)
            x = F.pad(x, (ga, gb, 0, 0) if last else (0, 0, ga, gb), mode="reflect")
            a -= ga; b -= gb
        return x

    return _axis(_axis(img, pad_l, pad_r, True), pad_t, pad_b, False)


class Runner:
    def __init__(self, args, model, postprocessor, input_size: int, device):
        import torch
        self.args = args
        self.model = model.to(device)
        self.pp = postprocessor.to(device)
        self.device = device
        self.input_size = input_size
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        self.B = int(args.batch_size)
        self.amp_dtype = {"none": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.amp]
        if args.filter_size > args.patch_size or (args.patch_size - args.filter_size) % 2:
            raise ValueError("--filter-size must be <= --patch-size and (patch - filter) must be even")
        if args.step_size > args.filter_size:
            print(f"[warn] --step-size {args.step_size} > --filter-size {args.filter_size}: ownership regions leave gaps "
                  f"({args.step_size - args.filter_size} px between tiles)")
        if args.step_size < args.filter_size and args.tile_merge_nms is None:
            raise ValueError(f"--step-size {args.step_size} < --filter-size {args.filter_size}: ownership regions overlap by "
                             f"{args.filter_size - args.step_size} px -> duplicate detections; use step == filter or set --tile-merge-nms")
        if args.tiling == "off":
            print("[warn] --tiling off: one forward per image; the model only fires in the centre "
                  f"{args.filter_size}x{args.filter_size} of its {input_size} input -> partial coverage for images larger than "
                  f"{args.filter_size} px (use --tiling auto for full coverage; off is for eval parity with the trainer)")

    # -- core forward on a fixed-size batch (padded to B so flex_attention (dynamic=False) never recompiles)
    def _forward(self, x01, orig_sizes):
        """x01: [n,3,S,S] in [0,1] on device; orig_sizes: [n,2] (w,h). Returns lists (labels, boxes xyxy, scores, cls_scores) len n."""
        import torch
        n = x01.shape[0]
        if n < self.B:
            pad = x01[-1:].expand(self.B - n, -1, -1, -1)
            x01 = torch.cat([x01, pad], 0)
            orig_sizes = torch.cat([orig_sizes, orig_sizes[-1:].expand(self.B - n, -1)], 0)
        x = (x01 - self.mean) / self.std
        with torch.no_grad():
            if self.amp_dtype is not None:
                with torch.autocast("cuda", dtype=self.amp_dtype):
                    out = self.model(x)
                out = {k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()}
            else:
                out = self.model(x)
            labels, boxes, scores, cls_scores = self.pp(out, orig_sizes.to(out["pred_boxes"].dtype))
        return labels[:n], boxes[:n], scores[:n], cls_scores[:n]

    def warmup(self):
        import torch
        t0 = time.time()
        size = self.args.patch_size if self.args.tiling in ("auto", "on") else self.input_size
        x = torch.rand(self.B, 3, size, size, device=self.device)
        s = torch.tensor([[size, size]] * self.B, device=self.device, dtype=torch.float32)
        self._forward(x, s)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        print(f"[warmup] batch={self.B} size={size} done in {time.time() - t0:.1f}s (includes flex_attention compile)")

    # -- whole-image batch
    def run_whole(self, samples: List[dict]):
        import torch
        x = torch.stack([s["img"] for s in samples]).to(self.device, non_blocking=True)
        sizes = torch.tensor([[s["orig_w"], s["orig_h"]] for s in samples], device=self.device, dtype=torch.float32)
        labels, boxes, scores, cls = self._forward(x, sizes)
        return [dict(labels=l, boxes=b, scores=s, cls_scores=c, n_tiles=1) for l, b, s, c in zip(labels, boxes, scores, cls)]

    # -- one image as a centred grid of tiles at native scale (single=True: exactly one centred tile, --tiling off)
    def run_tiled(self, sample: dict, single: bool = False):
        import torch
        a = self.args
        T, S, Fs = a.patch_size, a.step_size, a.filter_size
        if single:
            T = self.input_size; Fs = T  # one forward at the model input size, whole tile owned
        m = (T - Fs) // 2
        img = sample["img"]  # [3,H,W] cpu
        _, H, W = img.shape
        oxs, oys = tile_origins(W, T, S, Fs, single), tile_origins(H, T, S, Fs, single)
        pad_l, pad_t = max(0, -oxs[0]), max(0, -oys[0])
        pad_r, pad_b = max(0, oxs[-1] + T - W), max(0, oys[-1] + T - H)
        img = img.to(self.device, non_blocking=True).unsqueeze(0)
        img = reflect_pad(img, pad_l, pad_r, pad_t, pad_b)[0]
        mode = "reflect"
        coords = [(ox, oy) for oy in oys for ox in oxs]
        sizes = torch.tensor([[T, T]] * self.B, device=self.device, dtype=torch.float32)
        L, Bx, Sc, C = [], [], [], []
        for i in range(0, len(coords), self.B):
            chunk = coords[i: i + self.B]
            tiles = torch.stack([img[:, oy + pad_t: oy + pad_t + T, ox + pad_l: ox + pad_l + T] for ox, oy in chunk])
            labels, boxes, scores, cls = self._forward(tiles, sizes[: len(chunk)])
            for (ox, oy), l, b, s, c in zip(chunk, labels, boxes, scores, cls):
                if len(b) == 0:
                    continue
                cx = (b[:, 0] + b[:, 2]) / 2
                cy = (b[:, 1] + b[:, 3]) / 2
                # centre must be inside this tile's ownership region (half-open) AND inside the real image
                keep = (cx >= m) & (cx < m + Fs) & (cy >= m) & (cy < m + Fs)
                keep &= (cx + ox >= 0) & (cx + ox < W) & (cy + oy >= 0) & (cy + oy < H)
                if not keep.any():
                    continue
                b = b[keep] + torch.tensor([ox, oy, ox, oy], device=b.device, dtype=b.dtype)
                b[:, 0::2] = b[:, 0::2].clamp(0, W)
                b[:, 1::2] = b[:, 1::2].clamp(0, H)
                L.append(l[keep]); Bx.append(b); Sc.append(s[keep]); C.append(c[keep])
        if L:
            labels = torch.cat(L); boxes = torch.cat(Bx); scores = torch.cat(Sc); cls = torch.cat(C)
            if a.tile_merge_nms is not None and len(boxes) > 1:
                import torchvision
                keep = torchvision.ops.nms(boxes.float(), scores.float(), float(a.tile_merge_nms))
                labels, boxes, scores, cls = labels[keep], boxes[keep], scores[keep], cls[keep]
        else:
            labels = torch.empty(0, dtype=torch.long, device=self.device)
            boxes = torch.empty(0, 4, device=self.device)
            scores = torch.empty(0, device=self.device)
            cls = torch.empty(0, self.pp.num_classes, device=self.device)
        return dict(labels=labels, boxes=boxes, scores=scores, cls_scores=cls, n_tiles=len(coords),
                    tile_grid=[len(oxs), len(oys)], pad=[pad_l, pad_t, pad_r, pad_b], pad_mode=mode)


# --------------------------------------------------------------------------------------- outputs
def draw_overlay(path: str, out_path: str, boxes, labels, scores, thr: float, class_names: List[str], title: str = ""):
    from PIL import Image, ImageDraw
    im = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(im)
    keep = scores >= thr
    n = 0
    for b, l in zip(boxes[keep].tolist(), labels[keep].tolist()):
        d.rectangle(b, outline=CLASS_COLORS.get(int(l), (255, 255, 255)), width=2)
        n += 1
    if title:
        d.rectangle([0, 0, 8 * len(title) + 8, 16], fill=(0, 0, 0))
        d.text((4, 2), title, fill=(255, 255, 255))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    im.save(out_path, quality=92)
    return n


def _write_outputs(args, meta, class_names, num_classes, images_out, predictions, csv_rows, failed, class_totals,
                   elapsed, gt_json, img_root, items, n_vis, vis_dir, partial: bool):
    sfx = args.output_suffix or ""
    categories = [{"id": i, "name": class_names[i] if i < len(class_names) else str(i)} for i in range(num_classes)]
    meta.update({"date": _dt.datetime.now().isoformat(timespec="seconds"), "input": os.path.abspath(args.input),
                 "img_root": img_root, "num_images": len(items), "num_images_done": len(images_out), "partial": partial,
                 "conf_threshold": args.conf_threshold, "include_non_tumor": args.include_non_tumor, "tiling": args.tiling,
                 "patch_size": args.patch_size, "step_size": args.step_size, "filter_size": args.filter_size,
                 "tile_merge_nms": args.tile_merge_nms, "amp": args.amp, "tf32": args.tf32, "batch_size": args.batch_size,
                 "elapsed_sec": round(elapsed, 1), "class_totals": dict(zip(class_names, class_totals)), "failed": failed,
                 "gt_json": os.path.abspath(gt_json) if gt_json else None,
                 "cli": " ".join(sys.argv), "note": "bbox = [x,y,w,h] in original image pixels; scores >= conf_threshold only"})
    pred_path = os.path.join(args.output_dir, f"predictions{sfx}.json")
    with open(pred_path, "w") as f:
        json.dump({"meta": meta, "categories": categories, "images": images_out, "annotations": predictions}, f)
    csv_path = os.path.join(args.output_dir, f"per_image_counts{sfx}.csv")
    with open(csv_path, "w", newline="") as f:
        fields = list(csv_rows[0].keys()) if csv_rows else ["file_name", "image_id", "width", "height", "mode", "n_tiles", "n_det"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(csv_rows)
    print("=" * 80)
    print(f"{'PARTIAL ' if partial else ''}images={len(images_out)}/{len(items)} failed={len(failed)} "
          f"detections(>= {args.conf_threshold})={len(predictions)} per-class={dict(zip(class_names, class_totals))} "
          f"time={elapsed / 60:.1f} min ({len(images_out) / max(elapsed, 1e-9):.2f} img/s)")
    print(f"-> {pred_path}\n-> {csv_path}" + (f"\n-> {vis_dir} ({n_vis} overlays)" if args.visualize else ""))


def main(argv=None):
    args = parse_args(argv)
    if args.gpu is not None and str(args.gpu) != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.path.insert(0, DET_ROOT)
    import torch
    torch.manual_seed(args.seed)
    apply_tf32_policy(args.tf32)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    items, img_root, listing_json = collect_inputs(args)
    gt_json = args.gt_json
    coco_gt, class_names, cat_map = None, [], {}
    if gt_json:
        coco_gt, class_names, cat_map = attach_gt_ids(items, gt_json, img_root)
    if args.class_names:
        class_names = [s.strip() for s in args.class_names.split(",")]

    model, postprocessor, cfg, input_size, meta = build_model_and_postprocessor(args)
    num_classes = meta["num_classes"]
    if not class_names:
        class_names = DEFAULT_CLASS_NAMES[:num_classes] if num_classes <= len(DEFAULT_CLASS_NAMES) else [str(i) for i in range(num_classes)]
    if len(class_names) != num_classes:
        print(f"[warn] {len(class_names)} class names for {num_classes} classes: {class_names}")
    runner = Runner(args, model, postprocessor, input_size, device)
    if not args.no_warmup:
        runner.warmup()

    os.makedirs(args.output_dir, exist_ok=True)
    sfx = args.output_suffix or ""
    vis_dir = os.path.join(args.output_dir, f"vis{sfx}")
    vis_thr = args.vis_score_threshold if args.vis_score_threshold is not None else args.conf_threshold

    if vis_thr < args.conf_threshold:
        print(f"[warn] --vis-score-threshold {vis_thr} < --conf-threshold {args.conf_threshold}: overlays use {args.conf_threshold}")
        vis_thr = args.conf_threshold
    ds = InferenceDataset(items, input_size, args.tiling, args.patch_size)
    # images may differ in size -> one image per fetch; tiles / same-size whole images are batched by the Runner
    loader_bs = 1
    loader = torch.utils.data.DataLoader(ds, batch_size=loader_bs, shuffle=False, num_workers=args.num_workers,
                                         collate_fn=_identity_collate, pin_memory=(device.type == "cuda"))

    evaluator = None
    if coco_gt is not None:
        from src.data.dataset.coco_eval_aitod import AitodCocoEvaluator  # same class/maxDets as the trainer
        evaluator = AitodCocoEvaluator(coco_gt, ["bbox"])

    predictions: List[dict] = []
    images_out: List[dict] = []
    csv_rows: List[dict] = []
    failed: List[dict] = []
    ann_id = 1
    n_vis = 0
    class_totals = [0] * num_classes
    t_start = time.time()
    n_done = 0
    pending: List[dict] = []  # whole-mode (resized) samples waiting to fill a batch

    def emit(sample: dict, res: dict):
        nonlocal ann_id, n_vis, n_done
        labels, boxes, scores, cls = res["labels"].cpu(), res["boxes"].cpu(), res["scores"].cpu(), res["cls_scores"].cpu()
        if evaluator is not None:  # raw postprocessor output, exactly what det_engine.evaluate feeds it
            evaluator.update({int(sample["image_id"]): {"boxes": boxes, "scores": scores, "labels": labels}})
        keep = scores >= args.conf_threshold
        if not args.include_non_tumor:
            keep &= labels != 0
        labels, boxes, scores, cls = labels[keep], boxes[keep], scores[keep], cls[keep]
        counts = [int((labels == c).sum()) for c in range(num_classes)]
        for c in range(num_classes):
            class_totals[c] += counts[c]
        img_id = int(sample["image_id"])
        images_out.append({"id": img_id, "file_name": sample["file_name"], "width": sample["orig_w"], "height": sample["orig_h"],
                           "mode": sample["mode"], "n_tiles": res.get("n_tiles", 1),
                           **({"tile_grid": res["tile_grid"], "pad": res["pad"]} if "tile_grid" in res else {})})
        for i in range(len(scores)):
            x1, y1, x2, y2 = boxes[i].tolist()
            a = {"id": ann_id, "image_id": img_id, "category_id": int(labels[i]),
                 "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                 "area": round((x2 - x1) * (y2 - y1), 2), "score": round(float(scores[i]), 5), "iscrowd": 0}
            if not args.no_class_scores:
                a["class_scores"] = [round(v, 5) for v in cls[i].tolist()]
            predictions.append(a)
            ann_id += 1
        row = {"file_name": sample["file_name"], "image_id": img_id, "width": sample["orig_w"], "height": sample["orig_h"],
               "mode": sample["mode"], "n_tiles": res.get("n_tiles", 1), "n_det": int(len(scores))}
        for c in range(num_classes):
            row[f"n_{class_names[c] if c < len(class_names) else c}"] = counts[c]
        row["max_score"] = round(float(scores.max()), 4) if len(scores) else 0.0
        csv_rows.append(row)
        if args.visualize and (args.vis_limit is None or n_vis < args.vis_limit):
            rel = os.path.splitext(sample["file_name"])[0] + ".jpg"
            draw_overlay(sample["path"], os.path.join(vis_dir, rel), boxes, labels, scores, vis_thr, class_names,
                         title=f"{os.path.basename(sample['file_name'])}  n={len(scores)}")
            n_vis += 1
        n_done += 1
        if n_done % 50 == 0 or n_done == len(items):
            el = time.time() - t_start
            print(f"[{n_done}/{len(items)}] {n_done / el:.2f} img/s  elapsed {el / 60:.1f} min  "
                  f"totals={dict(zip(class_names, class_totals))}", flush=True)

    def flush_pending():
        nonlocal pending
        if pending:
            for s, r in zip(pending, runner.run_whole(pending)):
                emit(s, r)
            pending = []

    try:
        for batch in loader:
            for s in batch:
                if s["mode"] == "error":
                    print(f"[warn] skipping unreadable image {s['path']}: {s['error']}")
                    failed.append({"file_name": s["file_name"], "path": s["path"], "error": s["error"]})
                    n_done += 1
                elif s["mode"] == "tile":
                    flush_pending()
                    emit(s, runner.run_tiled(s))
                elif s["mode"] == "single":
                    flush_pending()
                    emit(s, runner.run_tiled(s, single=True))
                else:  # whole (resized to input_size) -> batched
                    pending.append(s)
                    if len(pending) == args.batch_size:
                        flush_pending()
        flush_pending()
    except BaseException as e:  # keep what was computed so far, then re-raise
        print(f"[error] aborted after {n_done}/{len(items)} images: {type(e).__name__}: {e} -> writing partial outputs")
        failed.append({"file_name": None, "path": None, "error": f"aborted: {type(e).__name__}: {e}"})
        _write_outputs(args, meta, class_names, num_classes, images_out, predictions, csv_rows, failed, class_totals,
                       time.time() - t_start, gt_json, img_root, items, n_vis, vis_dir, partial=True)
        raise
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.time() - t_start

    _write_outputs(args, meta, class_names, num_classes, images_out, predictions, csv_rows, failed, class_totals,
                   elapsed, gt_json, img_root, items, n_vis, vis_dir, partial=False)

    if evaluator is not None:
        print("=" * 80 + "\nCOCO eval (AitodCocoEvaluator, maxDets [500,1000,2000], raw postprocessor output):")
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()
        stats = evaluator.coco_eval["bbox"].stats.tolist()
        eval_out = {"gt_json": os.path.abspath(gt_json), "num_images": len(images_out), "checkpoint": meta["checkpoint"],
                    "weights": meta["weights"], "coco_eval_bbox": stats, "AP": stats[0], "AP50": stats[1]}
        ep = os.path.join(args.output_dir, f"coco_eval{sfx}.json")
        with open(ep, "w") as f:
            json.dump(eval_out, f, indent=2)
        print(f"AP={stats[0]:.4f} AP50={stats[1]:.4f} -> {ep}")
    print("done.")


if __name__ == "__main__":
    main()
