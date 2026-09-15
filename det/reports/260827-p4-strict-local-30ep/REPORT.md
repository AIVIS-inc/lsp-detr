# LSP-DETR-T bbox port — P4 strict-local 30-epoch 학습 기록

작성 2026-08-27 (학습 종료 당일). 대상 런: `[combined]LSP-T_strict-local_H100x8_hf5class-ft_30ep`
런 디렉터리: `/home/work/.mnt/DET_RESULT/lsp_detr/[combined]LSP-T_strict-local_H100x8_hf5class-ft_30ep/`
(`det/output` → 이 디렉터리로 심링크). 이 리포트의 모든 수치는 그 디렉터리의 `log.txt`(epoch별 JSON),
`train_run.log`(iteration 로그 + COCO 출력), 체크포인트 mtime에서 `make_report.py`로 파싱한 것이며,
`log.txt`와 `train_run.log`의 COCO 출력이 ep1–29 전 항목에서 일치함을 스크립트가 assert로 확인했다.

## 1. 결론 요약

| | ep0 (hf-5class 초기화 직후, launch #2) | **ep29 최종 (= best_stg1)** | Δ |
|---|---:|---:|---:|
| AP@[.5:.95] | 0.429 | **0.6154** | +0.186 |
| AP50 | 0.804 | **0.8749** | +0.071 |
| AR@2000 | 0.497 | **0.6711** | +0.174 |
| train loss (epoch mean) | 12.49 | 9.04 | −3.44 |

- **val AP가 30 epoch 내내 단조 증가** — best 체크포인트가 매 epoch 갱신됐고 최종 `best_stg1.pth` = `checkpoint0029.pth` = `last.pth` (모두 `last_epoch=29`, model 432 + EMA 432 텐서, CPU 로드 검증).
- 과적합 징후 없음. 같은 레시피 계열인 ConvNeXt-Dome v1(ep9 피크 후 하락)·YOLO26(ep6 피크)과 대조적.
- 수렴 완료: 마지막 3 epoch ΔAP +0.08 / +0.03 / +0.10 pp, AP50은 ep25 이후 4 epoch 합계 +0.06 pp. **추가 학습 가치 낮음** (§6).
- 같은 데이터셋의 Dome 기준선(P5)은 아직 없어 "LSP가 Dome보다 나은가"는 이 런만으로 판정 불가.

## 2. 런 정의

| 항목 | 값 |
|---|---|
| 모델 | `LSPDetrDetection` — LSP-DETR-T(Swinv2-T 백본, HF `hf-5class` 스냅샷) 핵 분할 → 박스 탐지 포트 (`det/lsp_det/`) |
| 아암 | **strict-local** (`center_mode=strict-local`, `center_span_cells=1.0`, `wh_prior_px=[14,14]`) |
| 초기화 | hf-5class 432 텐서 중 418 로드(coverage 99.99 %), 신규 5,390 파라미터 = class_head(bias prior 0.01) + wh_head 최종층(zero init); point_output은 hf-5class 그대로; 백본 45 텐서 동결(embeddings + encoder.layers.0) |
| 데이터 | `combined_all_v1_bundle` 2-class(Tumor / Non-tumor). train = `/home/work/.mnt/combined_all_v1_bundle_derived/train_coco_areafix.json`(ki67_NET 192만 ann에 area/iscrowd 보강), val = 9,853장, **GT는 중앙 672² 안에만 존재** |
| 입력 | 학습 `RandomCropWithGridNoDummy` 1536², centre_gt 672(빈 crop dummy GT 없음) + ColorJitter/Flip/Blur + ImageNetNormalize; 평가 Resize 1536² |
| 배치 | 8 × H100 80GB, GPU당 1장 (total 8), `OMP_NUM_THREADS=1`, `DOME_MATCH_THREADS=6` |
| 옵티마이저 | AdamW lr 2.5e-4 / wd 1.25e-4, 백본 lr 1.25e-5(norm은 wd 0), betas (0.9, 0.999), warmup 0 |
| 스케줄 | MultiStepLR milestones [24, 30], γ 0.8 → **ep24부터 ×0.8** (백본 1.25e-5 → 1.0e-5) |
| 손실 | LSPCriterion: VFL + L1 + GIoU, final + aux 5층 (denoising·DeFE 없음) |
| 평가 | EMA 가중치, Dome AITOD COCO 평가기(top-2000 + class-agnostic NMS 0.7, score 0.01), 8-GPU 분산 평가 |
| 정밀도 | TF32 **on** (`--tf32 keep`; 머신 전역 `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`, Dome 런들과 동일 조건) |
| 환경 | torch 2.13.0+cu130, CUDA 13.0, NCCL 2.29.7, transformers 5.13.0, `/home/work/miniconda3/envs/dome` |
| seed | 0 |
| 명령 | `RESUME=<run>/last.pth setsid nohup bash det/scripts/dist_train_lsp.sh` → `torch.distributed.run --standalone --nproc_per_node=8 train.py -c configs/LSP-T-combined.yml --seed 0 --tf32 keep -r <last.pth> -u epoches=30 train_dataloader.total_batch_size=8 val_dataloader.total_batch_size=8 LSPDetrDetection.center_mode=strict-local` |

## 3. 타임라인·사건 이력

| 시각 (KST) | 사건 |
|---|---|
| 08-18 오전 | **launch #1** — 시작 직후 `KeyError 'area'` (ki67_NET train ann에 area/iscrowd 없음) → `train_coco_areafix.json` 파생 파일 생성, config 교체 |
| 08-18 ~10:13 → 15:25 | **launch #2** — ep0 학습 5:11:44 (2.83 s/it) + eval → ep0 AP 0.429 / AP50 0.804, `checkpoint0000.pth` 기록 |
| 08-18 15:43 | launch #2 **ENOSPC 사망** — 출력이 49 GB 홈 loop 디스크(`det/output/`)에 있었음; rank0 `torch.save` iostream error + rank4/6 `train_samples` makedirs 실패(`train_run.log:318-356`). `best_stg1.pth` 손상, `last.pth`·`checkpoint0000.pth`는 정상 |
| 08-18 16:14 | **launch #3 = RESUME** — 출력을 `/home/work/.mnt/DET_RESULT/lsp_detr/`로 이동 후 `last.pth`(ep0)에서 재개, `setsid nohup`으로 분리 실행. 로그는 `launcher_resume.log` + `train_run.log`(tee -a) |
| 08-19 12:15, 19:04 / 08-20 23:13 | CUDA caching-allocator OOM **경고** 15건 (7.7–8.9 GB 블록 할당 실패 → 캐시 해제 후 재시도 성공, W 레벨). 이후 재발 없음. rank5는 런 내내 ~78 GB 예약 상태 유지(캐시, 누수 아님) |
| 08-24 | 사용자 VSCode 창 종료 — 런은 PPID 1·자체 세션이라 영향 없음(13:45 점검으로 확인) |
| 08-25 22:05 | ep24 체크포인트 = LR drop 적용된 첫 epoch |
| 08-27 12:19 / 12:34 | `checkpoint0029.pth` / 최종 eval → `best_stg1.pth`(ep29). solver 보고 `Training time 8 days, 20:17:09` (launch #3, ep1–29) |

DDP `Grad strides do not match bucket view strides` UserWarning이 시작 시 5회 출력됨(성능 경고, 정확성 무관).

## 4. 시간·자원

- 학습 스텝 시간: ep0 2.83 s/it → ep1 3.43 → ep12 3.95 → ep24–29 4.00–4.07 s/it (fig5). epoch당 학습 5.2 h(ep0) / 6.3–7.5 h, 평가 9–13 min, 체크포인트 저장 포함 **epoch 간격 평균 7.2 h, 후반 7.4 h**.
- 스텝 시간 증가 원인은 GPU가 아니라 CPU Hungarian matcher(11,664 query × GT 수천)임 — 8 GPU 사용률이 0–100 %를 주기적으로 오가는 패턴(rank 간 matcher 대기 동기화). 학습 후반으로 갈수록 예측이 많아져 matcher 비용이 늘어난 것으로 해석. 외부 CPU/GPU 경합은 없었음(load avg ≈ 21 / 64 core, 전부 학습 자체).
- GPU 메모리: rank0 `max mem` 27.1 GB; 실 예약은 rank별 30–38 GB, rank5 78 GB.
- 디스크: 체크포인트 30개 × 716 MB + best + last ≈ **22.9 GB** (NFS `/home/work/.mnt`, 17 TB 여유).

## 5. 결과

### 5.1 그래프

| 파일 | 내용 |
|---|---|
| `fig1_train_loss.png` | 총 train loss — iteration(100 it마다 20-step 중앙값) + epoch 평균. 30 epoch 내내 단조 하강, ep24 LR drop에서 −0.085(평소 −0.03의 3배) |
| `fig2_loss_components.png` | VFL / GIoU / L1 (final layer vs aux layer 0). 세 항 모두 끝까지 감소; L1은 ep8 이후 0.027–0.029 바닥 근처 |
| `fig3_val_metrics.png` | val AP50 / AR@2000 / AP (ep0–29, EMA). AP는 매 epoch 상승, AP50은 ep14부터 0.870–0.871 정체 → ep25에 0.8743으로 이탈 |
| `fig4_epoch_deltas.png` | epoch별 ΔAP / ΔAP50 (pp). AP 이득은 ep1–5 +1.8–2.0 pp → ep10 +0.5 → ep20+ +0.1 수준; AP50은 ep14 이후 ep25(+0.34)를 제외하면 ±0.05 |
| `fig5_epoch_time.png` | epoch별 s/it |

### 5.2 epoch 테이블

lr = 백본 그룹 lr(log.txt `train_lr`). loss는 epoch 평균(ep0은 launch #2 iteration 로그의 러닝 평균). `AP_verytiny*`는 8-GPU 평가의 area-subset 값으로 **오염된 수치**(§7) — 추세 참고용으로만 기재. s/it·train h는 학습 구간만, eval min은 검증 소요, ckpt time은 `checkpointNNNN.pth` mtime.

| ep | lr | train loss | VFL | GIoU | L1 | AP | AP50 | AP_verytiny* | AR@2000 | ΔAP (pp) | s/it | train h | eval min | ckpt time |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 1.25e-05 | 12.485 |  |  |  | 0.4290 | 0.8040 | 0.2460 | 0.4970 |  | 2.83 | 5.20 | 13 | 08-18 15:25 |
| 1 | 1.25e-05 | 11.008 | 1.348 | 0.423 | 0.0327 | 0.4477 | 0.8216 | 0.2622 | 0.5147 | +1.87 | 3.43 | 6.29 | 13 | 08-18 22:34 |
| 2 | 1.25e-05 | 10.666 | 1.308 | 0.399 | 0.0310 | 0.4653 | 0.8304 | 0.2797 | 0.5308 | +1.77 | 3.56 | 6.53 | 9 | 08-19 05:24 |
| 3 | 1.25e-05 | 10.428 | 1.278 | 0.385 | 0.0299 | 0.4854 | 0.8414 | 0.3017 | 0.5494 | +2.01 | 3.49 | 6.40 | 9 | 08-19 12:04 |
| 4 | 1.25e-05 | 10.270 | 1.258 | 0.374 | 0.0296 | 0.5047 | 0.8511 | 0.3219 | 0.5664 | +1.93 | 3.59 | 6.59 | 9 | 08-19 18:54 |
| 5 | 1.25e-05 | 10.157 | 1.243 | 0.366 | 0.0294 | 0.5226 | 0.8525 | 0.3401 | 0.5819 | +1.79 | 3.67 | 6.73 | 9 | 08-20 01:52 |
| 6 | 1.25e-05 | 10.033 | 1.227 | 0.358 | 0.0289 | 0.5357 | 0.8571 | 0.3557 | 0.5944 | +1.31 | 3.67 | 6.73 | 9 | 08-20 08:50 |
| 7 | 1.25e-05 | 9.943 | 1.215 | 0.353 | 0.0286 | 0.5496 | 0.8617 | 0.3709 | 0.6062 | +1.39 | 3.75 | 6.88 | 10 | 08-20 15:58 |
| 8 | 1.25e-05 | 9.884 | 1.209 | 0.347 | 0.0283 | 0.5602 | 0.8621 | 0.3857 | 0.6171 | +1.06 | 3.71 | 6.80 | 10 | 08-20 23:01 |
| 9 | 1.25e-05 | 9.799 | 1.196 | 0.343 | 0.0282 | 0.5673 | 0.8618 | 0.3964 | 0.6245 | +0.71 | 3.81 | 6.98 | 10 | 08-21 06:16 |
| 10 | 1.25e-05 | 9.744 | 1.188 | 0.340 | 0.0285 | 0.5728 | 0.8649 | 0.4046 | 0.6307 | +0.54 | 3.83 | 7.02 | 9 | 08-21 13:32 |
| 11 | 1.25e-05 | 9.683 | 1.180 | 0.337 | 0.0282 | 0.5780 | 0.8653 | 0.4115 | 0.6351 | +0.52 | 3.81 | 6.98 | 10 | 08-21 20:46 |
| 12 | 1.25e-05 | 9.646 | 1.175 | 0.334 | 0.0282 | 0.5826 | 0.8659 | 0.4141 | 0.6389 | +0.46 | 3.95 | 7.24 | 9 | 08-22 04:15 |
| 13 | 1.25e-05 | 9.588 | 1.168 | 0.332 | 0.0281 | 0.5858 | 0.8663 | 0.4177 | 0.6426 | +0.33 | 3.90 | 7.15 | 10 | 08-22 11:39 |
| 14 | 1.25e-05 | 9.536 | 1.161 | 0.329 | 0.0277 | 0.5900 | 0.8697 | 0.4218 | 0.6454 | +0.42 | 3.84 | 7.03 | 10 | 08-22 18:55 |
| 15 | 1.25e-05 | 9.518 | 1.159 | 0.328 | 0.0280 | 0.5910 | 0.8698 | 0.4255 | 0.6475 | +0.10 | 3.92 | 7.19 | 10 | 08-23 02:22 |
| 16 | 1.25e-05 | 9.469 | 1.152 | 0.326 | 0.0279 | 0.5943 | 0.8700 | 0.4306 | 0.6506 | +0.34 | 3.92 | 7.18 | 10 | 08-23 09:48 |
| 17 | 1.25e-05 | 9.433 | 1.148 | 0.324 | 0.0280 | 0.5966 | 0.8699 | 0.4337 | 0.6533 | +0.23 | 3.90 | 7.15 | 10 | 08-23 17:12 |
| 18 | 1.25e-05 | 9.389 | 1.143 | 0.322 | 0.0276 | 0.5991 | 0.8704 | 0.4370 | 0.6559 | +0.24 | 3.88 | 7.11 | 10 | 08-24 00:33 |
| 19 | 1.25e-05 | 9.362 | 1.140 | 0.321 | 0.0278 | 0.6015 | 0.8710 | 0.4379 | 0.6581 | +0.24 | 3.95 | 7.24 | 10 | 08-24 08:02 |
| 20 | 1.25e-05 | 9.329 | 1.136 | 0.320 | 0.0277 | 0.6032 | 0.8709 | 0.4394 | 0.6595 | +0.17 | 4.01 | 7.35 | 10 | 08-24 15:38 |
| 21 | 1.25e-05 | 9.294 | 1.133 | 0.319 | 0.0277 | 0.6039 | 0.8709 | 0.4412 | 0.6608 | +0.07 | 3.95 | 7.25 | 9 | 08-24 23:08 |
| 22 | 1.25e-05 | 9.272 | 1.128 | 0.319 | 0.0279 | 0.6048 | 0.8710 | 0.4434 | 0.6623 | +0.10 | 4.02 | 7.37 | 10 | 08-25 06:45 |
| 23 | 1.25e-05 | 9.245 | 1.126 | 0.317 | 0.0278 | 0.6085 | 0.8708 | 0.4461 | 0.6639 | +0.36 | 4.02 | 7.37 | 10 | 08-25 14:23 |
| 24 | 1.00e-05 | 9.160 | 1.114 | 0.313 | 0.0278 | 0.6093 | 0.8709 | 0.4487 | 0.6652 | +0.09 | 4.07 | 7.46 | 10 | 08-25 22:05 |
| 25 | 1.00e-05 | 9.132 | 1.111 | 0.311 | 0.0276 | 0.6116 | 0.8743 | 0.4497 | 0.6668 | +0.23 | 4.02 | 7.37 | 10 | 08-26 05:42 |
| 26 | 1.00e-05 | 9.106 | 1.107 | 0.310 | 0.0276 | 0.6133 | 0.8746 | 0.4523 | 0.6685 | +0.17 | 4.07 | 7.46 | 10 | 08-26 13:24 |
| 27 | 1.00e-05 | 9.082 | 1.106 | 0.309 | 0.0276 | 0.6141 | 0.8745 | 0.4539 | 0.6696 | +0.08 | 4.07 | 7.46 | 10 | 08-26 21:07 |
| 28 | 1.00e-05 | 9.061 | 1.102 | 0.309 | 0.0274 | 0.6144 | 0.8746 | 0.4544 | 0.6702 | +0.03 | 4.02 | 7.38 | 10 | 08-27 04:45 |
| 29 | 1.00e-05 | 9.044 | 1.102 | 0.307 | 0.0273 | 0.6154 | 0.8749 | 0.4553 | 0.6711 | +0.10 | 4.00 | 7.33 | 10 | 08-27 12:19 |

전체 13개 COCO stat과 iteration 단위 loss는 `epoch_metrics.csv`, `iter_loss.csv` 참조.

### 5.3 최종 COCO 출력 (ep29 EMA, 8-GPU 평가 원문)

```
AP 0.615  AP50 0.875  AR@500 0.550  AR@1000 0.650  AR@2000 0.671
area-subset (오염, §7): AP vt 0.455 / tiny 0.575 / small 0.664 / medium 0.547 · AR vt 0.506 / tiny 0.633 / small 0.722 / medium 0.630
```

## 6. 학습 추이 해석

1. **세 단계**: (a) ep0–8 급상승 — AP +1.1–2.0 pp/epoch, AP50 0.804 → 0.862 (새 class/wh head가 자리잡는 구간); (b) ep9–23 완만 — AP +0.1–0.7 pp/epoch, AP50 0.862 → 0.871 후 ep14부터 정체; (c) ep24–29 LR drop 이후 — AP50이 10 epoch 만에 처음 움직여 0.8743–0.8749, AP +0.03–0.23 pp/epoch.
2. **AP 상승 vs AP50 정체**: 후반 이득은 IoU 0.5에서의 탐지가 아니라 더 높은 IoU 문턱에서의 정밀도(박스 정합)에서 나왔다. 14 px 안팎의 핵 박스라 IoU ≥ 0.75 구간은 본질적으로 어렵고, 실무 지표로는 AP50(있냐/없냐)이 더 적절.
3. **train loss는 마지막까지 직선 하강(−0.023/epoch)** 하는데 val 이득은 0에 수렴 → train–val 갭이 벌어지기 시작하는 단계. 아직 val 하락(과적합)은 아니지만 같은 LR로 계속 돌려도 val로 전이되는 몫은 계속 줄어든다.
4. **LR 감소에만 반응**: 10 epoch 동안 유일한 AP50 움직임이 ep24 drop 직후(ep25 +0.34 pp)였다. 더 짜낼 여지가 있다면 epoch 연장이 아니라 LR annealing(예: 5 epoch cosine 1e-5 → 1e-6)이며, 기대 이득은 AP +0.2–0.5 pp / AP50 +0.1–0.3 pp, 비용 ~1.5일. 포화 곡선 외삽: ep24–29 구간 피팅 시 점근 AP 0.616(여지 +0.05 pp), ep15–29 전체 피팅 시 0.632(+1.6 pp, 낙관). **추천: 여기서 마감**하고 GPU를 P5 Dome 기준선 / movable-reference 아암에 사용.
5. 30 epoch·milestone [24, 30] 예산은 이 세팅에 거의 정확히 맞았다(ep29 = best, 낭비 epoch 없음).

## 7. 주의사항 (수치 인용 전 확인)

- **area-subset 통계(verytiny/tiny/small/medium AP·AR)는 8-GPU 평가에서 오염됨** — `DistributedSampler` 패딩(9,853 → 9,856) + `coco_eval_aitod.merge()`가 `eval_imgs`를 dedup하지 않아 (class, area) 블록이 어긋남. 전체 AP/AP50/AR@2000은 ≤ 0.001 차이로 정확. 정확한 subset 값은 `det/inference.py TILING=off`(단일 프로세스)로 재평가해야 함(ep3 기준 verytiny AP 0.302 → 실제 0.191이었음).
- **이 평가기의 13개 stat에 AP75는 없다** (순서: AP, AP50, AP_vt, AP_tiny, AP_small, AP_medium, AR@500/1000/2000, AR_vt/tiny/small/medium). 학습 중 점검 대화(08-25~27)에서 index 2를 "AP75"로 표기한 것은 오기 — 실제로는 AP_verytiny(오염 통계)였다. 본 리포트의 `epoch_metrics.csv`/표는 정정된 라벨을 쓴다.
- val GT가 중앙 672²에만 있으므로 모델은 1536 입력의 중앙에서만 발화하도록 학습됨 → 실제 추론은 `det/inference.py TILING=auto`(stride 672 타일링) 필수. 이미지 리사이즈 금지(작은 이미지는 reflect padding).
- best 선택 기준은 **AP**(`best_stat coco_eval_bbox[0]`). test_eval 킷의 다른 대상들은 AP50 선택인 경우가 있어 혼동 주의.
- TF32가 켜진 상태로 학습·평가됨(Dome 런들과 동일). 순수 FP32 재현이 필요하면 `--tf32 off`.
- `log.txt`에는 ep0 행이 없다(launch #2가 ENOSPC로 죽으면서 미기록). ep0 수치는 `train_run.log`의 COCO 출력에서 복원.
- 체크포인트에는 아암 정보(center_mode / wh_prior)가 없다 — `init_report.json`이 유일한 기록. resume·inference 시 반드시 함께 보관 (POST_TRAINING_TODO §2).
- `_get_match_pool` import는 Dome working-tree의 미커밋 패치에 의존 (POST_TRAINING_TODO §1) — 깨끗한 Dome checkout에서는 `import lsp_det`가 실패한다.

## 8. 산출물

런 디렉터리 (`/home/work/.mnt/DET_RESULT/lsp_detr/[combined]LSP-T_strict-local_H100x8_hf5class-ft_30ep/`):

| 파일 | 설명 |
|---|---|
| `best_stg1.pth` | **배포/평가용** — ep29 EMA (AP 0.6154), 716 MB |
| `last.pth`, `checkpoint0000–0029.pth` | 매 epoch 체크포인트(optimizer/EMA/scheduler 포함), 각 716 MB |
| `init_report.json` | 아암·wh_prior·hf-5class 로드 리포트 — 체크포인트와 항상 함께 보관 |
| `log.txt` | epoch별 JSON (ep1–29) |
| `train_run.log` / `launcher_resume.log` | launch #2 + #3 전체 stdout (iteration 로그, COCO 출력) / launch #3만 |
| `launcher_launch2.log` | launch #2 stdout (ENOSPC 사망 기록) |
| `train_samples/`, `val_samples/`, `summary/`, `eval/` | 시각화 샘플(5 epoch 간격), tensorboard, 평가 부산물 |
| `report/` | 이 리포트 사본 |

리포트 디렉터리 (`det/reports/260827-p4-strict-local-30ep/`): `REPORT.md`(이 문서), `epoch_metrics.csv`, `epoch_table.md`, `iter_loss.csv`, `summary.json`, `fig1–5*.png`, `make_report.py`(재생성 스크립트), `report_standalone.html`(그래프 내장·외부 의존 없음, 브라우저로 바로 여는 단일 파일 리포트), `build_html.py`.

## 9. 다음 단계

POST_TRAINING_TODO.md 순서대로: §1 Dome 의존성 고정(`_get_match_pool` fallback + 패치 diff 동봉) → §2 resume 아암 가드(**movable-reference 런 전 필수**) → `det/inference.py TILING=off`로 ep29 단일 프로세스 재평가(정확한 area-subset 수치) → P5 Dome 기준선(같은 bundle, `train_coco_areafix.json`, dummy 제거, top-2000) 또는 movable-reference 아암 런. test_eval 킷(`run_test_dumps.sh`)도 GPU 제약 없이 실행 가능.
