# LSP-DETR-T strict-local P4 run — hyper-parameters (as actually applied)

Run: `[combined]LSP-T_strict-local_H100x8_hf5class-ft_30ep` (launch #2 2026-08-18 10:11 → ENOSPC after epoch 0;
launch #3 = resume from `last.pth` 2026-08-18 16:14, epoch 1 onwards).
Output: `/home/work/.mnt/DET_RESULT/lsp_detr/[combined]LSP-T_strict-local_H100x8_hf5class-ft_30ep/` (`det/output` → symlink).
Sources: resolved cfg printed at launch (`launcher_resume.log`), `configs/LSP-T-combined.yml`,
`configs/include/lsp_swinv2.yml`, `configs/dataset/combined_tnt_detection.yml`, `scripts/dist_train_lsp.sh`, `init_report.json`.
The skeleton is the Dome-M HER2 recipe: `runtime.yml` / `dome/include/dataloader.yml` / `dome/include/optimizer.yml`
are included verbatim from `AIVIS-DETECTION/AIVIS-Dome-DETR/configs/`; only the items listed in §8 differ.

## 1. Batch / schedule
| item | value |
|---|---|
| train batch | **8 total** (1 per GPU × 8 H100), `drop_last: True` |
| val batch | 8 total (1 per GPU) |
| epochs | 30 (`epoches=30` via CLI); 6,601 iter/epoch = 52,802 crops ÷ 8 |
| LR scheduler | MultiStepLR, milestones [24, 30], gamma 0.8 (one ×0.8 drop at epoch 24) |
| warmup | none (`lr_warmup_scheduler.warmup_duration: 0`) |
| grad clip | `clip_max_norm: 0.1` |
| precision | FP32 (`use_amp: False`); TF32 ON (`--tf32 keep` = machine default `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`, `float32_matmul_precision=high`) |
| EMA | `ModelEMA` decay 0.9999, warmups 1000 steps, start 0 — validation / best use EMA weights |
| seed | 0 |
| checkpoint | every epoch (`checkpoint_freq: 1`) → `checkpointNNNN.pth`, `last.pth`, `best_stg1.pth` |
| eval | every epoch (`eval_freq` default 1) |
| sync_bn | True (no BN layers in this model — no effect) |
| find_unused_parameters | False |

## 2. Optimizer — AdamW, betas (0.9, 0.999), base lr 2.5e-4, base wd 1.25e-4
Four mutually exclusive groups (asserted by `lsp_det/optim_audit.py` at every launch):

| group | regex | scope | lr | wd | tensors | params |
|---|---|---|---|---|---|---|
| 0 | `^(?=.*backbone)(?!.*norm\|bn).*$` | backbone (Swinv2) non-norm | 1.25e-5 | 1.25e-4 | 152 | 27,248,772 |
| 1 | `^(?=.*backbone)(?=.*norm).*$` | backbone norm | 1.25e-5 | 0 | 44 | 19,200 |
| 2 | `^(?!.*backbone)(?=.*norm).*$` | non-backbone norm | 2.5e-4 | 0 | 38 | 14,592 |
| 3 | default / unmatched | decoder, heads, sampling | 2.5e-4 | 1.25e-4 | 153 | 17,288,474 |

Trainable: 387 tensors / 44,571,038 params. Frozen: 45 tensors / 308,646 params
(`backbone_freeze_at: 1` + `backbone_freeze_patch_embed: True` = patch-embed + Swin stage 1 = `encoder.layers.0`).
Bias is *not* split into a no-decay group (Dome parity).

## 3. Model — `LSPDetrDetection`, arm `strict-local`
| item | value |
|---|---|
| backbone | HF Swinv2-Tiny, patch 4, `backbone_window_size: 16`, `backbone_image_size: 1536` (window/shift geometry built for 1536, not 256), `backbone_drop_path_rate: 0.1` |
| dim / heads | 384 / 12 |
| query grid | `query_block_size: 14.2222` (= 256/18) → 108 × 108 = **11,664 queries** at 1536 px |
| feature_levels | [2, 1, 0, 2, 1, 0] |
| self-STA | kernel 3, q_tile 3, kv_tile 3 |
| cross-STA | kernel 5, q_tile 3, kv_tile 8 / 4 / 2 (per level) |
| center_mode | `strict-local` (local-sigmoid coordinates, point head loaded from checkpoint); `movable_span_cells: 3.0` unused in this arm |
| wh prior | `wh_prior_px: [14.0, 14.0]` (train_coco per-axis bbox median, `configs/wh_prior.json`); wh output init zero |
| feature_sampling_fixed | False (snapshot pixel-coordinate `grid_sample` behaviour) |
| num_classes | 2 (0 = Non-tumor, 1 = Tumor); class head new, bias prior 0.01 |
| pretrained | `/home/work/tksong/lsp-detr/hf-5class/model.safetensors` — 418/432 tensors loaded, 44,874,294 params (99.99 %), 14 source tensors dropped, 5,390 new params |
| input norm | `expect_imagenet_norm: True`, `input_norm_check: warn` |

## 4. Loss / matcher — `LSPCriterion` (Dome criterion restricted to final + 5 aux)
| item | value |
|---|---|
| losses | `['vfl', 'boxes']` |
| weight_dict | loss_vfl 3, loss_bbox (L1) 3, loss_giou 1 |
| VFL | alpha 0.75, gamma 2.0 |
| use_uni_set | True (union match over final + aux with `num_boxes_go`, Dome parity) |
| matcher | HungarianMatcher (scipy), cost_class 3, cost_bbox 3, cost_giou 1, focal alpha 0.25 / gamma 2.0 |
| matcher threads | `DOME_MATCH_THREADS=6` (one thread per aux layer; images inside a batch are solved sequentially) |
| use_focal_loss | True |

## 5. Data / augmentation
| item | value |
|---|---|
| train | `MultiCropDatasetWrapper(CocoDetection)`; img `/home/work/.mnt/combined_all_v1_bundle`; ann `/home/work/.mnt/combined_all_v1_bundle_derived/train_coco_areafix.json` (18,678 images: 16,384 small ×1 crop, 22 medium ×3, 2,272 large ×16 → 52,802 crops), `crop_size 1536`, `crops_per_image 16`, `grid_selection_strategy random`, `shuffle True`, `num_workers 4`, `persistent_workers True` |
| train transforms | RandomCropWithGridNoDummy(1536, center_gt_size 672) → ColorJitter(b 0.1, c 0.1, s 0.1, h 0.05) → RandomHorizontalFlip 0.5 → RandomVerticalFlip 0.5 → SanitizeBoundingBoxes(min_size 2) → ConvertPILImage float32 → ConvertBoxes cxcywh normalized → RandomGaussianBlur(p 0.25, k [3,5], σ [0.1,1.0]) → ImageNetNormalize |
| val | `CocoDetection`, ann `val_coco.json` (9,853 images), Resize [1536,1536] → ConvertPILImage → ImageNetNormalize, `shuffle False`, `num_workers 2` |
| collate | `BatchImageCollateFunction` |
| eval_spatial_size | [1536, 1536] |

## 6. Post-processing / evaluation
| item | value |
|---|---|
| DomePostProcessor | `num_top_queries 2000`, `use_nms True`, `nms_iou_threshold 0.7`, `nms_score_threshold 0.01`, `class_agnostic True`, `keep_topk_after_nms null` |
| evaluator | `AitodCocoEvaluator`, iou_types ['bbox'], maxDets [500, 1000, 2000], areas all / verytiny / tiny / small / medium |
| best selection | `coco_eval_bbox[0]` (AP@[.5:.95]) on EMA weights → `best_stg1.pth` |
| aux metric | top-4000 via `scripts/eval_topk.py` (not run by the solver) |

## 7. Launch environment (`scripts/dist_train_lsp.sh`)
`torch.distributed.run --standalone --nproc_per_node=8`, `CUDA_VISIBLE_DEVICES=0..7`, `OMP_NUM_THREADS=1`,
`DOME_MATCH_THREADS=6`, `PYTORCH_CUDA_ALLOC_CONF` unset, `NCCL_DEBUG=WARN`, `SAVE_INTERMEDIATE_VISUALIZE_RESULT=False`,
python `/home/work/miniconda3/envs/dome/bin/python`.
CLI overrides: `epoches=30 train_dataloader.total_batch_size=8 val_dataloader.total_batch_size=8 LSPDetrDetection.center_mode=strict-local`,
`--seed 0 --tf32 keep`, launch #3 additionally `-r <run>/last.pth` (`RESUME=` env).
Visualisation: `visualization_epoch_interval 5`, 1 batch per GPU, `visualization_save_rank0_only False`.

## 8. Intended differences vs the Dome-M 2-class baseline (P5)
1. Model: HGNetv2 + Dome encoder/decoder → Swinv2-T + STA decoder (`lsp_swinv2.yml`).
2. `RandomCropWithGrid` → `RandomCropWithGridNoDummy` (no fake 3×3 class-0 GT on empty crops).
3. `ImageNetNormalize` appended as the last transform (checkpoint preprocessor).
4. Optimizer regex: 4 groups instead of Dome's 2 (backbone LayerNorm must not be caught at base LR).
5. Train annotation file: `train_coco_areafix.json` (area = w·h, iscrowd = 0 for the 66 ki67_NET images) — the P5 Dome baseline must use the same file.
Everything else (batch 8, 30 ep, lr 2.5e-4 / backbone 1.25e-5, MultiStep [24,30]×0.8, EMA 0.9999, clip 0.1, VFL/L1/GIoU 3/3/1,
matcher 3/3/1, top-2000 + NMS 0.7, TF32-on FP32) is the Dome recipe.

## 9. Observed so far (val, EMA)
| epoch | train loss | AP | AP50 | AR@2000 | AP verytiny / tiny / small | epoch time |
|---|---|---|---|---|---|---|
| 0 | 12.48 | 0.429 | 0.804 | 0.497 | 0.246 / 0.374 / 0.500 | 5 h 12 m (2.83 s/it) |
| 1 | 11.01 | 0.448 | 0.822 | 0.515 | 0.262 / 0.397 / 0.517 | 6 h 17 m (3.43 s/it) |
| 2 | 10.67 | 0.465 | 0.830 | 0.531 | 0.280 / 0.417 / 0.532 | 6 h 32 m (3.56 s/it) |

Per-class at epoch 2 (from `eval/latest.pth`): Non-tumor AP 0.439 / AP50 0.825 / AR 0.508; Tumor AP 0.492 / AP50 0.835 / AR 0.553.
Step time is CPU-Hungarian-bound (GPU fwd ≈ 0.6 s of 2.5–4.7 s); ~26 GB GPU peak; ≈ 6.5 h/epoch → 30 ep ≈ 8 days.
