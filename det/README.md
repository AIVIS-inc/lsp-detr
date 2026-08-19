# LSP-DETR → bbox detector port (Dome-DETR recipe, hf-5class fine-tune)

Everything for the port lives in this folder; the Dome framework
(`/home/work/tksong/AIVIS-DETECTION/AIVIS-Dome-DETR`) is imported as an external dependency and is **not modified**.
Single source of truth for design decisions: `../FINETUNE_STRATEGY.md` (2026-08-18).

```
det/
├── train.py                     entry point (Dome train.py flow + LSP registration + optimizer audit + --tf32 policy)
├── inference.py / run_inference.sh   inference (image/dir/COCO json → COCO-style predictions, tiling, overlays, COCO eval)
├── lsp_det/
│   ├── __init__.py              bootstraps Dome onto sys.path ($DOME_ROOT), imports/registers everything below
│   ├── lsp_trunk.py             STA decoder / Cayley-STRING / FeatureSampling / point-wh-class heads (from hf-5class/modeling.py)
│   ├── lsp_detr_det.py          LSPDetrDetection (Swinv2-T offline from config + trunk; encoder.use_defe stub; freeze policy)
│   ├── checkpoint.py            controlled hf-5class load (remap + exact missing/dropped assert)
│   ├── lsp_criterion.py         LSPCriterion(DomeCriterion): final+5 aux, vfl+boxes, uni_set boxes
│   ├── transforms.py            RandomCropWithGridNoDummy, ImageNetNormalize (double-normalize guard)
│   └── optim_audit.py           param-group audit (exclusive regexes, LR/WD policy, frozen set)
├── configs/
│   ├── dataset/combined_tnt_detection.yml   2-class bundle (copy of Dome coco_detection.yml + crop_size)
│   ├── include/lsp_swinv2.yml               model/criterion block (counterpart of dome_hgnetv2.yml)
│   ├── LSP-T-combined.yml                   top-level (include chain = Dome-M-AITOD.yml with 2 swaps)
│   └── wh_prior.json                        P0 bbox median statistics
├── scripts/  dist_train_lsp.sh, cfg_diff_vs_dome.py, smoke_optim_groups.py, smoke_p3_{1,2,3}_*.py,
│             make_smoke_subsets.py, eval_topk.py, compute_wh_prior.py
├── tests/    pytest unit tests (Cayley stale-P, criterion empty batches, transforms)
├── logs/     smoke outputs (p*_*.log/json)
└── IMPLEMENTATION_LOG.md
```

## Environment
* conda env `dome`: `/home/work/miniconda3/envs/dome/bin/python` (py3.11, torch 2.13.0+cu130, torchvision 0.28, transformers 5.13.0)
* extra deps installed into that env for this port: `einops==0.8.1`, `pytest`
* no network needed: Swinv2 is built from `Swinv2Config` and all weights (backbone included) come from
  `../hf-5class/model.safetensors`
* Dome path: auto-detected as `../../AIVIS-DETECTION/AIVIS-Dome-DETR` (sibling checkout); override with `DOME_ROOT=/path`.
  The config include chain uses the same relative path (`det/configs/LSP-T-combined.yml`).
* **TF32**: this machine sets `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`, i.e. TF32 matmul is ON by default for every torch
  process (the reference Dome runs included). `det/train.py --tf32 keep|off|on` (default `keep`). The RoPE angle
  computation in the port is TF32-immune (elementwise) either way.

## Commands
```bash
cd /home/work/tksong/lsp-detr
PY=/home/work/miniconda3/envs/dome/bin/python

# unit tests (CPU)
cd det && CUDA_VISIBLE_DEVICES="" $PY -m pytest tests -q -p no:warnings; cd ..

# P2 gates
CUDA_VISIBLE_DEVICES="" $PY det/scripts/cfg_diff_vs_dome.py          # resolved cfg vs Dome train_run.log:32
CUDA_VISIBLE_DEVICES="" $PY det/scripts/smoke_optim_groups.py --names   # optimizer groups

# P3 smokes
DOME_MATCH_THREADS=6 OMP_NUM_THREADS=1 $PY det/scripts/smoke_p3_1_single_gpu.py --device cuda:0
$PY det/scripts/smoke_p3_2_parity.py --tf32 off
$PY det/scripts/smoke_p3_3_overfit.py --arm strict-local --device cuda:0 --steps 600
$PY det/scripts/smoke_p3_3_overfit.py --arm movable-reference --device cuda:1 --steps 600
$PY det/scripts/make_smoke_subsets.py
NUM_GPUS=2 GPU_IDS=2,3 END_EPOCHS=2 OUTPUT_DIR=logs/p3_4_ddp2 \
  EXTRA_UPDATES="train_dataloader.dataset.dataset.ann_file=$PWD/det/logs/smoke_data/train_subset.json val_dataloader.dataset.ann_file=$PWD/det/logs/smoke_data/val_subset.json train_dataloader.num_workers=2 val_dataloader.num_workers=2" \
  bash det/scripts/dist_train_lsp.sh

# P4 (full run, 8 GPU, 30 ep, seed 0) - DO NOT start without approval
ARM=strict-local bash det/scripts/dist_train_lsp.sh
ARM=movable-reference bash det/scripts/dist_train_lsp.sh
# env knobs: NUM_GPUS GPU_IDS END_EPOCHS SEED OUTPUT_DIR TF32=keep|off EXTRA_UPDATES="key=val ..." MASTER_PORT

# auxiliary production metric (top-4000 / maxDets 4000), separate output name
$PY det/scripts/eval_topk.py -c det/configs/LSP-T-combined.yml -r <ckpt.pth> --output-dir <dir> --topk 4000 --split val
```

## Inference (`det/inference.py`, launcher `det/run_inference.sh`)
Counterpart of the Dome `run_*.sh` tools for the LSP arm; the eval path is reproduced 1:1 (/255 → ImageNet mean/std →
`LSPDetrDetection` (hf-5class init disabled, EMA weights, arm taken from the run's `init_report.json`) → `DomePostProcessor`
from the yml, top-2000 / class-agnostic NMS 0.7).
```bash
cd det
./run_inference.sh /path/to/img_or_dir_or_coco.json                       # defaults: RESUME=<run>/best_stg1.pth, GPU=0, TILING=auto
GPU=4 LIMIT=20 VISUALIZE=True ./run_inference.sh /home/work/.mnt/combined_all_v1_bundle/val/breast_HER2
# val split + COCO eval == trainer's log.txt numbers (AitodCocoEvaluator on the raw postprocessor output; TILING=off!)
TILING=off INPUT=/home/work/.mnt/combined_all_v1_bundle/val_coco.json IMG_FOLDER=/home/work/.mnt/combined_all_v1_bundle \
GT_JSON=/home/work/.mnt/combined_all_v1_bundle/val_coco.json GPU=4 ./run_inference.sh
# knobs (env): RESUME WEIGHTS=ema|model CENTER_MODE GPU BATCH_SIZE CONF_THRESHOLD NMS_IOU_THRESHOLD NUM_TOP_QUERIES INCLUDE_NON_TUMOR
#   TILING=auto|on|off PATCH_SIZE=1536 STEP_SIZE=672 FILTER_SIZE=672 TILE_MERGE_NMS VISUALIZE VIS_LIMIT AMP TF32
#   OUTPUT_DIR OUTPUT_SUFFIX LIMIT IMG_FOLDER GT_JSON EXTRA_ARGS; extra flags may also follow INPUT on the command line
```
* Inputs: image / directory (recursive) / `*.txt` path list / COCO json (`images[].file_name` + `IMG_FOLDER`).
* **The model only fires in the centre 672² of a 1536 input** (GT constraint of the recipe), and training never rescaled
  images (small ones are reflect-padded to 1536 at native scale). Therefore
  * `TILING=auto` (default, production): every image is run at **native scale** as a *centred* grid of 1536 tiles, stride
    672, centre-672 ownership (half-open; centre must also lie inside the real image), symmetric reflect padding
    (iterative, like `_handle_small_image`) — full coverage, no cross-tile NMS unless `TILE_MERGE_NMS`. Images ≤ 672 px are a
    single centred tile (a 512 image sits at [512,1024) exactly as in training); a 1536 image becomes 3×3 tiles.
  * `TILING=off` (eval parity only): one forward per image — 1536 inputs as-is (== val transform), smaller ones centred with
    reflect padding, larger ones resized (val `Resize` semantics); coverage is centre-only, the script warns.
* Arm safety: `center_mode` / `movable_span_cells` / `wh_prior_px` are CLI overrides at training time and leave no trace in
  the checkpoint (identical tensor shapes), so the script reads `init_report.json` next to the checkpoint and **raises** if the
  built model disagrees (`--allow-arm-mismatch` / `CENTER_MODE=` to override). `scripts/eval_topk.py` has no such guard.
* Outputs in `OUTPUT_DIR` (default `det/output/inference/<run>_<ckpt>_<time>/`, i.e. on the .mnt disk):
  `predictions{sfx}.json` (COCO-style: meta (incl. resolved arm, failed images, partial flag) / categories / images (mode,
  tile grid, pads) / annotations with `bbox` xywh in original pixels, `score`, `class_scores`; scores ≥ `CONF_THRESHOLD`
  only), `per_image_counts{sfx}.csv`, `vis{sfx}/` overlays (blue = Non-tumor, red = Tumor), `coco_eval{sfx}.json` (with
  `GT_JSON`), `inference{sfx}.log`. Unreadable images are skipped and listed; on an exception the partial outputs are written.
* Batches are padded to a fixed `BATCH_SIZE` because `flex_attention` is compiled with `dynamic=False` (each new batch size
  recompiles, ~30 s); the first call compiles (~40 s warm-up). ~4 img/s at 1536 (single forward) on a shared H100.
* `AMP=none` (default) matches the training numerics; `bf16`/`fp16` autocast work (±1–2 boxes / image on the smoke set).
* Verified 2026-08-19 on `best_stg1.pth` (ep3): `TILING=off` on the full val split reproduces log.txt (see IMPLEMENTATION_LOG);
  smoke on 4096/3375/512 train images in `auto` mode (49/36/1 tiles) with visually correct, seam-free overlays.

## Config knobs (LSPDetrDetection, `det/configs/include/lsp_swinv2.yml`)
`center_mode` strict-local|movable-reference (`movable_span_cells` 3.0), `wh_prior_px` [14,14], `pretrained`,
`pretrained_arm`, `backbone_image_size` 1536, `backbone_freeze_at` 1 (+`backbone_freeze_patch_embed`),
`backbone_drop_path_rate` 0.1, `feature_sampling_fixed` False, `expect_imagenet_norm`/`input_norm_check`.
