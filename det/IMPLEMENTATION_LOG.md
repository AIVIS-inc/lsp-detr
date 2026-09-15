# LSP-DETR bbox port — implementation log (2026-08-18)

Basis: `../FINETUNE_STRATEGY.md` (2026-08-18) overrides `AIVIS-DETECTION/docs/260817-lsp-detr-port-plan.md`.
Workspace rule: all new code in `det/`; Dome repo (`AIVIS-Dome-DETR`) import-only, **0 files modified**.

## Timeline / decisions

### P0
* env: dome conda (py3.11.9, torch 2.13.0+cu130, torchvision 0.28.0, transformers 5.13.0). Installed `einops==0.8.1`, `pytest`.
* GPUs: 8×H100 80GB all idle at start (V3-B0 not running).
* checkpoint audit: `hf-5class/model.safetensors` 432 tensors / 45,024,444 params; key groups backbone 241 / layers 114 /
  point_head 36 / radial 36 / class 2 / feature_sampling 3 (matches §3).
* wh prior (`det/scripts/compute_wh_prior.py` → `det/configs/wh_prior.json`): 43,953,630 train boxes,
  **median w = 14.0 px, median h = 14.0 px** (log 2.639); p05/p95 = 9/24; class0 14/14, class1 16/16; per-source 12.5–16.
  Config uses `wh_prior_px: [14.0, 14.0]` (log(7.5) not used anywhere).
* `det/train.py`: Dome train.py flow; `import lsp_det` bootstraps `$DOME_ROOT` (default sibling checkout) onto sys.path,
  imports `src.core/src.zoo/src.data/src.optim`, then registers the LSP components in the same registry
  (`LSPDetrDetection`, `LSPCriterion`, `RandomCropWithGridNoDummy`, `ImageNetNormalize`) — no `src/zoo/__init__.py`
  edit needed (the plan's "from . import lsp" is replaced by external registration).

### P1 (model port) — gate: import + CPU construction + checkpoint mismatch 0 → PASS
* `lsp_trunk.py` from `hf-5class/modeling.py` (attribute names kept 1:1: `layers/point_head/class_head`,
  `pe.freqs`, `pe.parametrizations.S.original`, `feature_sampling.reduction/norm`). Changes:
  - `CayleySTRING.P` cached_property → plain method recomputed per eval call (§5.2). Test `tests/test_cayley_string.py`
    reproduces the snapshot staleness and checks eval == train(solve) after S updates.
  - `flex_attention = torch.compile(..., dynamic=False)`; `torch._dynamo.config.cache_size_limit` raised to 64.
  - block-mask lru_cache keyed on device string (DDP-safe).
  - **RoPE angle computed elementwise instead of `einsum`** (see TF32 finding below; mathematically identical).
  - `radial_distances_head` → `wh_head` (MLP 384→384→384→2, output zero-init), `class_head=Linear(384,2)` bias prior 0.01
    (`-log(99)`), shared across layers as in the snapshot; look-forward-twice for centres and log-wh; log-wh clamp
    [-6, 9] only inside the box computation.
  - centre policy: `relative_to_absolute_pos(..., span)`; span=1 strict-local (snapshot), span=3 movable-reference
    (centre may move over [-1,+2) cells; sigmoid(0)=0.5 still maps to the cell centre).
  - `FeatureSampling` bug (pixel coords into grid_sample) preserved by default; `feature_sampling_fixed` switch (§4.3).
* `lsp_detr_det.py`: `LSPDetrDetection` (@register, `__share__=[num_classes]`), Swinv2-T via
  `Swinv2Config(image_size=1536, patch4, embed 96, depths 2/2/6/2, heads 3/6/12/24, window 16, drop_path 0.1)` +
  `AutoBackbone.from_config` (offline). Backbone keys 241/241 loaded. Trunk attribute `decoder`; `encoder=_EncoderStub()`
  (plain object, `use_defe=False`, no params). Freeze: `embeddings.*` + `encoder.layers.0.*` (45 tensors, 308,646 params).
  Input-normalisation check (one-time warn if the first batch lies in [0,1]). No `.deploy()`.
* `checkpoint.py`: remap `decode_head.→decoder.`, `radial_distances_head.{i}.{0,2}→wh_head.{i}.{0,2}`; drop
  `radial_distances_head.{i}.4`, `class_head` (+ `point_head.{i}.4` for movable arm, re-zeroed). Exact-set asserts on
  missing (14 keys) / dropped (14 keys) + shape check + zero-init/bias-prior re-assert. Result strict-local:
  **418/432 tensors, 44,874,294 params loaded (99.99 % of 44,879,684), 5,390 new** — identical to §3.1 numbers.
  Movable arm: 406/432, 44,869,674 loaded, 10,010 new.
* `lsp_criterion.py`: `LSPCriterion(DomeCriterion)`, forward re-implemented for `[final] + 5 aux` only, threads via
  Dome `_get_match_pool`, uni_set boxes with `num_boxes_go`, VFL per-layer match with `num_boxes`, `(outputs, targets,
  **kwargs)`; `nan_to_num` on losses kept for Dome parity (note: like Dome, a NaN loss becomes 0 rather than aborting).
  Tests: keys = 3 + 15 aux, mixed empty, all-empty (L1/GIoU exactly 0, VFL finite > 0), registry creation.
* `transforms.py`: `RandomCropWithGridNoDummy` (only `_add_dummy_box` overridden → identity; covers all 3 call sites);
  `ImageNetNormalize(T.Normalize)` with double-application guard (flag on the produced tensor + input range check,
  images only — BoundingBoxes pass through). Tests: empty crop stays [0,4]/[0]; Dome original adds a class-0 3×3 box, ours not;
  full train pipeline + val pipeline normalise exactly once; boxes untouched.
* `optim_audit.py`: exclusive-assignment + LR/WD policy + frozen-prefix asserts; called from `det/train.py` before fit.

### P2 (configs/launcher) — gate: cfg diff BAD 0 → PASS
* `configs/dataset/combined_tnt_detection.yml` (Dome coco_detection.yml + num_classes 2 + bundle paths +
  **`crop_size: 1536`** on `MultiCropDatasetWrapper` — required because the wrapper auto-detects crop_size only from a
  transform literally named `RandomCropWithGrid`; with `RandomCropWithGridNoDummy` it would silently fall back to 2048).
* `configs/include/lsp_swinv2.yml` (model/criterion/postprocessor names, `use_focal_loss: True`, model block, criterion block).
* `configs/LSP-T-combined.yml`: same include chain as Dome-M-AITOD.yml with `runtime.yml / include/dataloader.yml /
  include/optimizer.yml` referenced *from the Dome repo* (relative path); top-level = Dome-M's overrides
  (checkpoint_freq 1, visualisation keys, eval_spatial_size, DomePostProcessor top-2000/NMS, MultiStepLR [24,30] 0.8,
  warmup 0, dead `epochs: 30`) + transforms ops restated (lists don't merge) with NoDummy + ImageNetNormalize last +
  optimizer 3 regex groups (`backbone∧¬norm` 1.25e-5 | `backbone∧norm` 1.25e-5/wd0 | `¬backbone∧norm` wd0 | default).
* `scripts/dist_train_lsp.sh`: clone of dist_train.sh (env `OMP_NUM_THREADS=1`, `DOME_MATCH_THREADS=6`, unset
  `PYTORCH_CUDA_ALLOC_CONF`, NCCL WARN), entry `det/train.py`, env-overridable knobs, `--tf32 keep` default.
* `scripts/cfg_diff_vs_dome.py` (simulates the launcher CLI incl. argparse `use_amp=False`): 217 vs 201 flattened keys,
  139 identical, 127 differing — **all** in the allowed classes (model blocks, model/criterion name, dataset paths/classes,
  NoDummy+crop_size, ImageNetNormalize ×2, optimizer groups [1..2], run identity). Recipe keys asserted identical:
  use_ema/ema/clip 0.1/use_amp False/use_focal_loss/epoches 30/milestones/lr/wd/backbone lr/top-2000/workers/batch/
  checkpoint_freq/sync_bn/find_unused_parameters/warmup 0. Log: `logs/p2_cfg_diff.log`.
* `scripts/smoke_optim_groups.py`: 387 trainable tensors → groups 152 / 44 / 38 / 153 (27.25M / 19.2K / 14.6K / 17.29M);
  backbone LayerNorms (44) at 1.25e-5 wd 0 (Dome regex trap avoided); `feature_sampling.norm` and decoder norms wd 0;
  frozen set exactly embeddings + layers.0. Log: `logs/p2_optim_groups.log`.

### P3-1 single GPU (`scripts/smoke_p3_1_single_gpu.py`, real config, val split densest images) → PASS
* shapes: pred_logits [1,11664,2], pred_boxes [1,11664,4], 5 aux; all finite; wh > 0; unused params 0.
* 2,739-GT image, B=1: **peak 24.5 GB**, step 3.4–3.7 s = fwd 0.57 + criterion **2.6–3.0 (Hungarian on 11,664×2,739 ×6)** +
  bwd 0.18 + opt 0.01. B=2 (one empty): peak 46.0 GB. compile warm-up 49 s (first step), +11.6 s for the B=2 shapes.
* all-empty batch: L1/GIoU 0, VFL finite (loss 3843 because `num_boxes` clamps to 1 → mean over 11,664 queries ×Q/1;
  same normalisation as Dome; grad clip 0.1 caps the update anyway).
* eval path: postprocessor top-2000 + NMS → 1900 dets, `all_class_scores` [K,2].
* loss decreased 25.1 → 15.3 over 6 steps on the same 4 images; wh median 14.0 → 16.9 px.

### P3-2 parity at 256 (`scripts/smoke_p3_2_parity.py`) → PASS (pure FP32)
* HF snapshot loaded via a shim `load_backbone` (offline Swinv2 image_size=256), strict load 432/432; port built with
  `backbone_image_size=256` for identical Swin geometry.
* pure FP32 (`--tf32 off`): backbone 4 stages **bit-exact**; embeddings max|Δ| 8.1e-5 (rel 1.3e-5); centres 2.7e-4 px.
* cached-P: snapshot `P` differs from P(current S) by 0.27 after an S update yet its eval output moves 0.0 (stale);
  port `P()` vs float64 1.9e-7, eval output moves and equals train(solve) path within 4.5e-5.
* geometry note: at 256 input, a 1536-config backbone differs from the 256-config one in stage3/4 (window/shift by
  design) — irrelevant for training at 1536.
* **TF32 finding**: `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1` is set machine-wide (NGC image) → `allow_tf32=True`,
  precision `high` in every torch process, including the reference Dome HER2 run (no code sets it; the "TF32 off"
  belief was code-based). Under TF32 the snapshot's `einsum` RoPE angle has max error 0.175 rad at 256 px and
  **1.4 rad at 1536 px** (10-bit mantissa on pixel coordinates); the port's elementwise angle is exact (2.4e-4 rad).
  Under machine-default TF32, snapshot(einsum) vs port(elementwise) differ by 0.24 in embeddings / 0.65 px in centres at
  256 — that is the TF32 noise the snapshot itself suffers. `det/train.py --tf32` (keep|off|on) added; default keep.

### P3-4 DDP / EMA / val ×2 (`scripts/dist_train_lsp.sh`, 2 GPUs, subsets 24 train / 16 val, 2 epochs) → PASS
* launched through the real launcher (`NUM_GPUS=2 GPU_IDS=2,3 END_EPOCHS=2`, ann_file overrides via `EXTRA_UPDATES`);
  `[tf32] mode=keep ... allow_tf32=True`, optimizer audit printed on rank 0, hf5class init line, FLOPs skip warning
  (expected, no `.deploy()`), MultiCropDistributedSampler 24 crops, val DistributedSampler.
* 12 iters/epoch/rank; no DDP unused-parameter error (`find_unused_parameters=False`, static compile + DDP OK);
  epoch 0 val mAP 0.0044 / AP50 0.0176 → epoch 1 val **0.0152 / 0.0380** (second validation reflects updated weights);
  `log.txt` 2 lines, `checkpoint0000/0001.pth`, `last.pth`, `best_stg1.pth`, `eval/latest.pth`, TensorBoard summary,
  train/val sample PNGs written (colours off because inputs are ImageNet-normalised — cosmetic).
* checkpoint audit: EMA `updates` 12 → 24; EMA weights ep0→ep1 max|Δ| 3.1e-3 (model 3.1e-3), EMA≠model (6.9e-6, decay ramp);
  frozen `backbone.embeddings.*` unchanged (Δ 0.0); `parametrizations.S.original` keys present in `model`/`ema`;
  optimizer state groups (152 / 44 / 38 / 153) with lr 1.25e-5 / 1.25e-5 / 2.5e-4 / 2.5e-4 and wd 1.25e-4 / 0 / 0 / 1.25e-4.
* 8-GPU repeat (`NUM_GPUS=8`, 1 epoch): 3 iters/rank (9.8 s/it incl. compile), val 2 images/rank gathered across ranks,
  AP 0.0054 / AP50 0.0203, checkpoint written, no errors; 48 matcher threads (8×6) fine. Log `logs/p3_4_ddp8_launcher.log`.
* `det/train.py` now also writes `<output_dir>/init_report.json` (arm, centre span, point/wh/class init, wh prior, tf32
  state, frozen count, full hf5class load report) — checklist "point output 초기화 차이를 config와 checkpoint 메타에 기록".

### P3-3 overfit + arm ablation (`scripts/smoke_p3_3_overfit.py`, 10 val images 200<GT<1500, 5,386 GT, 13.5 % in
collision cells, recipe optimizer, clip 0.1, no EMA, deterministic transforms)
600 steps (60 passes/image), eval every 100 (official top-2000+NMS AP50, Hungarian matched-IoU median, greedy IoU≥0.5 recall):

| step | strict-local AP50 / mIoU / recall all / collision / non-coll | movable-reference AP50 / mIoU / recall all / collision / non-coll |
|---|---|---|
| 0   | 0.002 / 0.469 / 0.066 / 0.069 / 0.066 | 0.000 / 0.257 / 0.011 / 0.008 / 0.011 |
| 100 | 0.316 / 0.713 / 0.731 / 0.481 / 0.770 | 0.274 / 0.694 / 0.697 / 0.482 / 0.730 |
| 200 | 0.660 / 0.751 / 0.838 / 0.613 / 0.873 | 0.598 / 0.742 / 0.839 / 0.628 / 0.872 |
| 300 | 0.783 / 0.764 / 0.864 / 0.658 / 0.896 | 0.782 / 0.758 / 0.862 / 0.686 / 0.890 |
| 400 | 0.851 / 0.756 / 0.892 / 0.680 / 0.924 | 0.849 / 0.776 / 0.895 / 0.737 / 0.920 |
| 500 | 0.879 / 0.749 / 0.900 / 0.715 / 0.928 | 0.846 / 0.753 / 0.895 / 0.744 / 0.919 |
| 600 | **0.903** / 0.786 / 0.916 / **0.723** / 0.946 | **0.883** / 0.785 / 0.911 / **0.787** / 0.930 |

* Both arms fit steadily (loss ↓, matched IoU 0.47→0.79 / 0.26→0.79, AP50 still rising at 600) — pipeline/loss/matcher/
  postprocessor are consistent end-to-end. Neither reached the 0.95 gate within 600 steps → extended 1500-step runs
  (see below).
* Structural ceiling: on these 10 images strict-local (one query per 14.2 px cell) can never recover 372/5,386 GT
  (≥2 centres in one cell) → **max recall 0.931**, so AP50 ≥ 0.95 is unreachable for strict-local *by construction*;
  achieved 0.916 recall at 600 steps is already 98 % of that ceiling. movable-reference has no hard ceiling and leads on
  collision recall (0.787 vs 0.723) while trailing slightly on overall AP50 at 600 steps (its point outputs restart from zero).

## §9 implementation checklist (FINETUNE_STRATEGY.md)
| # | item | status / evidence |
|---|---|---|
| 1 | Swinv2 `image_size=1536`, backbone 241/241 strict compat | ✅ `build_swinv2_backbone(image_size=1536)`; controlled load 241 backbone keys, `logs/p3_2_parity.log` strict 432/432 on the snapshot |
| 2 | no `load_backbone`, use `AutoBackbone.from_config` | ✅ `lsp_detr_det.build_swinv2_backbone` (offline) |
| 3 | `einops==0.8.1` recorded as dependency | ✅ installed in dome env; README "Environment" |
| 4 | remap → expected missing/unexpected assert | ✅ `checkpoint.py` exact-set asserts (14 missing / 14 dropped; movable 26/26) |
| 5 | radial hidden .0/.2 → wh hidden; only class/wh outputs new | ✅ 44,874,294 loaded / 5,390 new |
| 6 | wh prior = train bbox median, no log(7.5) | ✅ (14.0, 14.0) px, `configs/wh_prior.json` |
| 7 | strict-local vs movable point-output init recorded in config + ckpt meta | ✅ `center_mode` in cfg, `init_report.json` per run |
| 8 | Cayley stale cache removed + train→eval test | ✅ `tests/test_cayley_string.py`, `logs/p3_2_parity.log` |
| 9 | ImageNet Normalize exactly once train and val | ✅ last op in both pipelines; double-apply guard; `tests/test_transforms.py` |
| 10 | `_add_dummy_box` path removed + all-empty batch loss test | ✅ `RandomCropWithGridNoDummy`; `tests/test_criterion_empty.py`; P3-1 all-empty step |
| 11 | LSPCriterion final+5 aux only, no FGL/DDF/CDN/DeFE keys | ✅ `lsp_criterion.py`; loss keys 3 + 15 aux |
| 12 | `encoder.use_defe=False` stub + registry import | ✅ `_EncoderStub`; registration via `lsp_det/__init__.py` (Dome untouched) |
| 13 | optimizer groups exclusive, backbone norm lr 1.25e-5 | ✅ `optim_audit.py` (asserted at every launch), `logs/p2_optim_groups.log` |
| 14 | official top-2000 vs production top-4000 saved under separate names | ✅ trainer = 2000; `scripts/eval_topk.py` → `eval_{split}_top4000.{pth,json}` (not yet run on a trained ckpt) |
| 15 | FLOPs skip warning normal, no `.deploy()` | ✅ warning observed in P3-4 logs; no deploy() |
| + | first-epoch empty-crop / empty-batch / per-class GT statistics (§6.1) | ✅ `LSPCriterion._stats_update`: per-epoch counters (crops, empty crops, all-empty batches, GT boxes, per-class), all-reduced and printed at the last step of every epoch (`[LSPCriterion][epoch N] ...`) — no Dome edit needed |

1500-step runs (eval every 300, same 10 images) — **gate PASS for both arms** (AP50 ≥ 0.95, matched IoU rising):

| step | strict-local AP50 / mIoU / recall all / collision / non-coll | movable-reference AP50 / mIoU / recall all / collision / non-coll |
|---|---|---|
| 900  | 0.948 / 0.834 / 0.958 / 0.828 / 0.979 | 0.934 / 0.802 / 0.947 / 0.865 / 0.960 |
| 1200 | 0.968 / 0.873 / 0.975 / 0.877 / 0.991 | 0.963 / 0.840 / 0.978 / 0.931 / 0.985 |
| 1500 | **0.974** / 0.840 / 0.981 / **0.902** / 0.994 | **0.979** / 0.849 / 0.987 / **0.957** / 0.991 |

* Correction to the "ceiling" note above: with IoU≥0.5 tolerance a query in an *empty neighbouring* cell can cover a
  GT whose centre lies in a collision cell, so strict-local recall is not hard-capped at 0.931; the one-per-cell limit
  shows up as the persistent **collision-recall gap (0.902 vs 0.957 at 1500; 0.877 vs 0.931 at 1200)** while
  non-collision recall is ~equal (0.994 vs 0.991).
* Reading for the arm decision (§4.2): both arms fit the data (AP50 0.97–0.98, mIoU 0.84–0.85 — 15-px boxes make
  mAP@[.5:.95] ≈ 0.65–0.72 the realistic overfit level); movable-reference (span 3 cells, point outputs zero re-init)
  converges a little slower early (worse at 300–900) but ends higher and recovers +5.5 pp of collision GT.
  Overall recall difference at 1500: 0.987 vs 0.981 (+0.6 pp) — collision GT are 13.5 % of this sample.
* Logs/JSON: `logs/p3_3_overfit_{strict-local,movable-reference}{,_1500}.{log,json}`.

## Dome repo modifications
**None.** (`git status` in AIVIS-DETECTION shows only pre-existing modifications — coco_detection.yml, dist_train.sh,
dome_criterion.py from earlier sessions — none newer than the strategy doc; `find -newer FINETUNE_STRATEGY.md` on
src/configs/train.py/dist_train.sh returns nothing.) The plan's `src/zoo/__init__.py` edit was avoided by external
registration through `det/lsp_det/__init__.py`.

## Open items / risks (for the P4 decision)
1. **TF32 policy**: machine default is TF32-on (Dome-actual). `--tf32 keep` (default) keeps Dome parity of the actual
   condition; `--tf32 off` matches the doc's wording (pure FP32, slower backbone). RoPE angles are exact either way now.
2. **Step time / matcher**: 11,664 queries × dense GT → 1.0–3.0 s/step per rank at 1 img/GPU (matcher-bound; fwd 0.57 s).
   6,600 it/epoch × 30 ep ≈ 198K steps → ~2.5–6 days depending on GT density (Dome-M was ~1.26 s/step on the same
   infra). GPU LAP (torch-linear-assignment) remains the reserve lever, not used to keep matcher parity.
3. `nan_to_num` in the criterion (Dome parity) turns a NaN loss into 0 silently; the engine only aborts if the *reduced*
   loss is non-finite. Consider `-u`-free monitoring of `Loss/*` in TensorBoard during P4.
4. Strict-local one-query-per-cell: collision recall gap ~5 pp on the overfit sample (13.5 % collision GT); the
   production candidate is movable-reference per §4.2, but the *first* comparison baseline stays strict-local.
5. Auxiliary top-4000 metric (`eval_topk.py`) is implemented but was not exercised on a trained checkpoint yet.
6. Empty-batch VFL magnitude (num_boxes clamps to 1 → loss ~3.8k on an all-empty *global* batch) is inherited from
   Dome's normalisation; grad clip 0.1 bounds the update. Per-rank empties are averaged out by the all-reduce.
7. Visualisation PNGs (`train_samples/`, `val_samples/`) look wrong-coloured because inputs are ImageNet-normalised
   (cosmetic; Dome's `save_samples` assumes [0,1]).

## P4 launch (2026-08-18, approved: ARM=strict-local, TF32=keep)
* Launch #1 crashed in the first DataLoader batch: `KeyError: 'area'` in Dome `ConvertCocoPolysToMask`
  (`obj["area"]` read unconditionally). Audit (`scripts/audit_area_keys.py`): train has **1,924,260 annotations on 66
  ki67_NET images without `area`/`iscrowd`** (val/test complete). The plan had noted the missing `iscrowd` (handled by Dome)
  but not `area`; smokes used val/subsets so never hit it.
* Fix (data-side, Dome untouched): `scripts/fix_train_area.py` writes
  `/home/work/.mnt/combined_all_v1_bundle_derived/train_coco_areafix.json` (area = bbox w·h, iscrowd = 0; images/ids/
  categories identical; original bundle file unmodified; README.txt in the derived dir). Dataset yml points at it; cfg diff
  still PASS (path = allowed diff); crop counts unchanged (52,802); ki67_NET images load. **The Dome P5 baseline must use
  the same derived file.**
* Launch #2 running: `det/output/[combined]LSP-T_strict-local_H100x8_hf5class-ft_30ep/` (train_run.log, init_report.json,
  log.txt per epoch, checkpoint*.pth, best_stg1.pth). First 100 iters: loss 20.7 → ~17.2 (avg), 6,601 it/epoch/rank.
* **Launch #2 died 2026-08-18 15:43 (ENOSPC on /home/work)** after epoch 0 (train 5:11:44 @ 2.83 s/it, val 12:47 +
  accumulate 188 s; ep0 EMA: AP 0.429 / AP50 0.804 / AR@2000 0.497). rank0 hit `No space left on device` while writing
  `best_stg1.pth` (368 MB partial, corrupt), the other ranks in `makedirs(train_epoch001_samples)`; all 8 procs exited.
  Root cause: `det/output/` lives on the 49 GB `/home/work` loop device (91 % used); with `checkpoint_freq: 1` the run
  needs ≥ 23 GB (30 × 716 MB + last + best) → it could never finish there. `last.pth` / `checkpoint0000.pth`
  (last_epoch=0) load fine. Fix: OUTPUT_DIR on `/home/work/.mnt/DET_RESULT/` (as the Dome/ConvNeXt runs) + resume via
  the new `RESUME=` env of `dist_train_lsp.sh` (`-r last.pth`, solver restarts at epoch 1). Realistic wall-clock is
  ~5.5 h/epoch → ~6.6 days for the remaining 29 epochs.
* **Launch #3 (resume) 2026-08-18 16:14**: run dir moved to `/home/work/.mnt/DET_RESULT/lsp_detr/[combined]LSP-T_strict-local_H100x8_hf5class-ft_30ep/`
  (`det/output` is now a symlink to `/home/work/.mnt/DET_RESULT/lsp_detr`; launcher default `OUTPUT_ROOT` points there; smoke ckpts
  `logs/p3_4_ddp{2,8}` moved to `.../lsp_detr/smoke/` with symlinks). Launched with `RESUME=<run>/last.pth` → log shows
  `Resume checkpoint from … / Load last_epoch`, training restarted at `Epoch: [1]` with loss 11.76 at iter 0 (≈ end-of-ep0
  level, i.e. weights/optimizer state restored). Logs: `train_run.log` (appended), `launcher_resume.log`, `launcher_launch2.log`
  (the crashed launch). Home disk after cleanup: 36/49 GB.

## Inference tooling (2026-08-19, written while P4 launch #3 trains; Dome untouched)
* New: `det/inference.py` + `det/run_inference.sh` (Dome `run_*.sh`-style launcher, env knobs; see README "Inference") and
  `tests/test_inference_geometry.py` (CPU). Modes: `TILING=auto` (production, full coverage: centred 1536-tile grid, stride
  672, centre-672 ownership, symmetric iterative reflect padding at native scale = training small-image path) and
  `TILING=off` (eval parity: one forward per image). Reads the arm (`center_mode`/`center_span_cells`/`wh_prior_px`) from the
  run's `init_report.json` and raises on mismatch (the checkpoint itself carries no arm; strict load cannot catch it).
* Adversarial review (4 lenses × 2 skeptics, 24 findings, 18 confirmed) fixed before verification: relative paths broken by
  `cd det/` in the launcher, no `python -u`, small images 3×-upsampled instead of reflect-padded (a 512 HER2 tile went from
  13 to 159 detections), step<filter double counting, GT-id trust, duplicate ids, unreadable image killing a run (now
  skipped + partial outputs written on any exception).
* **Verification** (`best_stg1.pth` = ep3, EMA):
  - full val (9,853 images, `TILING=off`, GPU shared with training, 46.7 min @ 3.5 img/s): AP 0.4846 / AP50 0.8414 /
    AR@2000 0.5489 vs log.txt 0.4854 / 0.8414 / 0.5494 → overall metrics match to ≤ 0.001.
  - `det/train.py --test-only` (the trainer's own single-process eval path) vs the script on a 395-image val subset:
    **all 13 COCO stats identical to 3 decimals** (AP 0.483, AP50 0.844, verytiny 0.193, tiny 0.439, small 0.548, medium 0.438,
    AR 0.456/0.532/0.548, 0.208/0.502/0.619/0.498).
  - tiling smoke on 4096² (49 tiles) / 3375² (36) / 512² (1) train images: overlays seam-free, boxes on nuclei, class colours
    follow DAB (ITGB6 core: 8.7k Non-tumor / 3.6k Tumor).
* **Side finding (Dome eval, not this port)**: log.txt *area-subset* stats of the 8-GPU run differ from the single-process
  evaluation (verytiny AP 0.302 vs 0.191, AR 0.333 vs 0.207, medium 0.444 vs 0.434) although AP/AP50/AR@2000 agree.
  Cause: `DistributedSampler` pads 9,853 → 9,856 (3 duplicate images) and Dome's `coco_eval_aitod.merge()` dedups
  `img_ids` (`np.unique`) but not `eval_imgs`, so the flat C++ evalImgs buffer (K×A×9856) is read with I = 9853 → each
  (class, area) block is shifted by 3·(block index) entries and reads a few whole-image entries of the previous block
  (verytiny GT are rare, so a handful of 'all'-area entries inflate it). Affects every 8-GPU log.txt on this val set
  (Dome/ConvNeXt baselines included); overall AP/AP50 shift ≤ 0.001. Trust the subset stats only from single-process evals
  (`det/inference.py` with GT_JSON, or `train.py --test-only` on one GPU).

## 2026-08-27 — P4 strict-local 30-epoch run COMPLETE

* Run `[combined]LSP-T_strict-local_H100x8_hf5class-ft_30ep` finished 2026-08-27 12:34 KST (launch #3 resume from ep0,
  solver "Training time 8 days, 20:17:09"; launch #2 had produced ep0 before dying of ENOSPC on 08-18). No NaN, no crash;
  15 benign CUDA caching-allocator OOM warnings on 08-19/20 (allocator retried), none after.
* **Final = best = ep29 EMA: AP 0.6154 / AP50 0.8749 / AR@2000 0.6711** (ep0: 0.429 / 0.804 / 0.497). Val AP rose every
  epoch; AP50 plateaued at 0.870–0.871 from ep14 and moved only after the ep24 LR ×0.8 milestone (ep25 +0.34 pp).
  Last-3-epoch ΔAP +0.08/+0.03/+0.10 pp → converged; not worth extending at this LR (analysis in the report §6).
* Pace: 2.83 s/it (ep0) drifting to 4.00–4.07 s/it (ep24–29), 7.2–7.5 h per epoch incl. ~10 min eval; CPU
  Hungarian matcher-bound. Checkpoints 30 × 716 MB + best + last ≈ 22.9 GB on the NFS mount.
* Full record (per-epoch CSV + markdown table, iteration-level loss CSV, 5 figures, timeline/incidents, caveats,
  regeneration script): `det/reports/260827-p4-strict-local-30ep/REPORT.md`; copy in `<run>/report/`.
* Correction: the Dome AITOD evaluator's 13 stats are AP, AP50, AP_verytiny, AP_tiny, AP_small, AP_medium, AR@500/1000/2000,
  AR_vt/tiny/small/medium — there is **no AP75**. Index 2 (0.4553 at ep29) is AP_verytiny, an 8-GPU merge()-polluted
  subset stat, not AP75 as it was called during the daily checks.
* Next: POST_TRAINING_TODO §1 (Dome dependency pin) → §2 (resume arm guard, required before movable-reference) →
  single-process re-eval of ep29 with `det/inference.py TILING=off` for true area-subset stats → P5 Dome baseline /
  movable-reference arm. GPUs are idle from 12:34.

## 2026-09-15 — `det` branch made self-contained (trains from a bare clone, no sibling repo)

* **Dome-DETR runtime vendored** into `det/third_party/dome/`: `src/` (91 modules, incl. the thread-pool Hungarian
  matcher patch of `dome_criterion.py` that AIVIS-DETECTION never committed and every det/ run trained with),
  `tools/{visualize_src_flatten,visualize_image_annotation,concatenate_images}.py` (the only `tools.*` modules `src/`
  imports), `configs/`, Apache-2.0 `LICENSE`; provenance (AIVIS-DETECTION@1e6a557 working tree, `diff -r` clean) and the
  re-sync recipe in `VENDORED.md`. `lsp_det/__init__.py` bootstraps that copy; `$DOME_ROOT` is now an override, not a
  requirement. Closes POST_TRAINING_TODO §1 — `lsp_criterion.py` additionally falls back to sequential matching
  (result-identical) if an override checkout lacks `_get_match_pool`.
* Config include chains (`LSP-T-combined.yml`, `LSP-T-her2.yml`) read the vendored `runtime.yml` /
  `dome/include/{dataloader,optimizer}.yml`; `lsp_swinv2.yml` `pretrained:` is repo-relative
  (`hf-5class/model.safetensors`, resolved against the lsp-detr root by `LSPDetrDetection.resolve_pretrained_path`,
  which raises a FileNotFoundError naming the fetch script) — POST_TRAINING_TODO §4.
* `scripts/fetch_hf5class.py`: the 180 MB initial checkpoint comes from HF hub `RationAI/LSP-DETR` at the **pinned**
  revision `a32176184e` (2025-07-09; sha256 `3f5437eb…`, 180,151,024 B — the revision whose config.json/modeling.py are
  the tracked `hf-5class/` files). The hub replaced `model.safetensors` on 2025-08-20 (`99b0d385…`, 180,178,896 B), so an
  unpinned download would NOT reproduce the P4/HER2 initialisation; the script verifies the checksum and is idempotent.
* `requirements.txt`: pinned versions of the `dome` env (torch 2.13.0+cu130, torchvision 0.28.0, transformers 5.13.0,
  faster-coco-eval-aitod 1.0.2 (PyPI), …), derived from the distributions actually imported while building the full
  training config; the two custom evaluator packages are on PyPI.
* `scripts/dist_train_lsp.sh`: python resolved like `run_inference.sh` (`PYTHON=` → training-box conda env → PATH),
  `OUTPUT_ROOT` defaults to `det/output` when the NFS dir is absent, dataset overrides
  `TRAIN_IMG_DIR/TRAIN_ANN/VAL_IMG_DIR/VAL_ANN` (→ `-u` updates, placed before `EXTRA_UPDATES`), `DRY_RUN=1` prints the
  resolved torchrun command, preflight for the config file and the hf-5class checkpoint. On the training box the
  resolved command is unchanged. `scripts/cfg_diff_vs_dome.py` now requires `--dome-log` (the reference Dome run's log
  is not part of the repo).
* Verified in a **fresh clone with no sibling checkout**, `DOME_ROOT` unset, CUDA hidden: `import lsp_det` resolves
  `src`/`tools` inside the clone; without the weights the model build and the launcher preflight fail with the fetch
  hint; `fetch_hf5class.py` downloads + verifies; 29 unit tests pass; `LSP-T-her2.yml` and `LSP-T-combined.yml` build
  model (418/432 tensors) / criterion / optimizer (4 groups, audit OK) / EMA / evaluator, nested dataset `-u` overrides
  land; `DRY_RUN=1` assembles the same command as before; every `src.*` submodule imports except the two dead paths
  that also fail in the original tree (`coco_eval_aitod_slow` → aitodpycocotools, `deformable_encoder` → unbuilt MSDA
  extension); `train.py --test-only -d cpu` runs the vendored solver → model forward (FlexAttention CPU forward) →
  postprocessor → AitodCocoEvaluator end-to-end on 2 val images (29.5 s/it on 8 threads, `eval.pth` written; AP≈0 as
  expected from the untrained new heads). A CPU *training* step is impossible — FlexAttention has no CPU backward
  (torch limitation, unchanged by the port); the attempt got through dataloader/transforms/backbone into the decoder
  before that error, and its first run caught a real gap (`det_engine.py` imports `tools.concatenate_images`), fixed
  by vendoring that module. No GPU was used: all 8 were busy with the HER2 run (ep19/30), which kept running from the
  untouched `/home/work/tksong/lsp-detr` working tree throughout (that tree is now behind `origin/det` and must be
  updated only after the run ends — its launcher bash process is still alive).
