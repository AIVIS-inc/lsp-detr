# LSP-DETR 객체탐지 파인튜닝 전략

작성 2026-08-17, 검토·갱신 2026-08-18.

마스터 플랜: `/home/work/tksong/AIVIS-DETECTION/docs/260817-lsp-detr-port-plan.md`

> **최종 판정: 구현 가능하며 `hf-5class` 전체 트렁크 파인튜닝을 권장한다.** 다만 Dome의 bbox head를 그대로 복사하는 작업이 아니라, LSP의 384차원 STA 디코더에 LSP-native `(cx, cy, log-w, log-h)` head를 붙이고 Dome의 데이터·매처·손실·후처리·학습 인프라를 재사용하는 포팅이다.
>
> **작업 장소는 이 레포가 아니다.** 실제 구현과 학습은 `/home/work/tksong/AIVIS-DETECTION/AIVIS-Dome-DETR`에서 한다. 이 레포의 Hydra/Lightning/rationai-mlkit/stardist 학습 스택은 사용하지 않는다. 원형 LSP 학습은 인스턴스 마스크가 필요한데 `combined_all_v1_bundle`에는 bbox만 있기 때문이다.

## 1. 목표와 범위

`hf-5class/`의 RationAI/LSP-DETR 5-class 사전학습 가중치를 초기값으로 사용해 LSP-DETR을 bbox 탐지기로 전환하고, `/home/work/.mnt/combined_all_v1_bundle`의 2개 클래스(0=Non-tumor, 1=Tumor)를 Dome-DETR HER2 런과 최대한 같은 레시피로 학습한다.

- 유지: Swinv2-Tiny 백본, FeatureSampling, 6층 STA self/cross attention, Cayley-STRING RoPE, look-forward-twice 중심점 정제.
- 교체: 64방향 radial head → 2차원 log-wh head, 6-class explicit-background head → 2-class sigmoid head.
- 재사용: Dome COCO 데이터 파이프라인, Hungarian matcher, VFL/L1/GIoU, EMA, optimizer/scheduler, DDP, bbox evaluator, NMS 후처리.
- 제외: 마스크/radial GT 생성, StarDist, PQ/AJI, DeFE, MWAS, CDN, FGL/DDF.
- 비교군: 동일 데이터와 동일 레시피로 학습한 Dome-M 2-class 모델. 기존 5-class Dome 체크포인트는 직접 비교군이 아니다.

## 2. 기존 마스터 플랜과 대조 결과

마스터 플랜의 Option A와 학습 인프라 재사용 방향은 유지한다. 아래 항목은 이번 실측 검토 결과로 수정하거나 추가하며, 이 문서의 내용이 마스터 플랜의 해당 항목보다 우선한다.

| 항목 | 기존 계획 | 갱신 결론 |
|---|---|---|
| 체크포인트 호환성 | P0에서 확인 예정 | `safetensors`/`.pth` 값 동일 확인. Swinv2 `image_size=1536` 모델과 백본 키·shape 241/241 일치 확인 |
| radial head | 36 tensors 전량 폐기 | 각 MLP의 hidden Linear `.0/.2` 24 tensors는 wh head에 이식 권장. 64-output `.4`만 폐기 |
| wh prior | `log(7.5)` | 7.5는 radial 반경 prior라 bbox 폭·높이에 부적절. train bbox의 `(median_w, median_h)`를 구해 `log` prior로 사용. 임시 기본값은 qbs와 비슷한 14–16 px 범위 |
| 분류 의미론 | K+1 softmax-focal로 기술 | 원본은 K+1 **sigmoid focal** + explicit no-object. 이식본은 K=2 sigmoid VFL + all-zero background. class head는 재초기화 |
| local query | LSP 정체성으로 그대로 유지 | parity arm에서는 유지하되, 동일 qbs cell 충돌이 약 8–9% 수준이라 production arm은 ±1-cell 또는 global movable reference를 별도 검증 |
| 빈 crop | 추가 작업 없음 | 현행 `RandomCropWithGrid`가 가짜 3×3 class-0 GT를 생성. 반드시 제거하고 empty target을 정상 처리 |
| Transformers API | 4.52→5.13 드리프트 리스크 | 구형 `load_backbone`은 실제 import 실패. 현행 `AutoBackbone.from_config` 사용. modern Swinv2와 체크포인트 shape 호환은 확인 완료 |
| Cayley-STRING | 그대로 이식 | HF 코드의 `cached_property P`는 첫 평가 뒤 갱신된 `S`를 반영하지 못할 수 있음. 평가 시 매번 재계산하도록 캐시 제거 |
| top-k | 2000 고정 | Dome parity 지표는 2000 유지. GT가 2000개를 넘는 val/test 이미지가 있으므로 production 보조 평가는 top-k/maxDets 4000도 산출 |

## 3. 체크포인트 실측 결과와 로딩 정책

`hf-5class/model.safetensors`와 `model_5class.pth`를 직접 비교했다.

- 두 파일 모두 state dict 432 tensors, 45,024,444 parameters이며 모든 tensor 값이 동일하다.
- `.pth`는 `state_dict`와 config를 담은 wrapper이고, 실제 로딩에는 안전하고 단순한 `model.safetensors`를 사용한다.
- point/radial 최종층은 0이 아니므로 초기화 파일이 아니라 실제 학습 완료 가중치다.
- 현재 Dome 환경에서 `Swinv2Config(image_size=1536, ...)` + `AutoBackbone.from_config`로 만든 백본과 체크포인트 `backbone.*`가 키·shape 241/241 전부 일치했다.

| 체크포인트 구성요소 | tensors | parameters | 처리 |
|---|---:|---:|---|
| `backbone.*` | 241 | 27,576,618 | 전부 로드 |
| `decode_head.layers.*` | 114 | 13,449,216 | `decoder.layers.*`로 remap 후 전부 로드 |
| `decode_head.point_head.*` | 36 | 1,778,700 | strict-local arm은 전부 로드 |
| `feature_sampling.*` | 3 | 295,680 | 전부 로드, 원본 동작 유지 |
| `decode_head.radial_distances_head.*` | 36 | 1,921,920 | hidden `.0/.2`는 wh head로 remap, output `.4`는 폐기 |
| `decode_head.class_head.*` | 2 | 2,310 | 폐기, `Linear(384, 2)` 신규 생성 |
| 합계 | 432 | 45,024,444 | — |

### 3.1 권장 초기화

1. `backbone.*`, `feature_sampling.*`, `decode_head.layers.*`, `decode_head.point_head.*`를 로드한다.
2. `decode_head.radial_distances_head.{i}.{0,2}.*`를 `decoder.wh_head.{i}.{0,2}.*`로 remap한다.
3. 각 `wh_head.{i}.4`는 `Linear(384, 2)`로 만들고 weight/bias를 0으로 초기화한다.
4. `class_head=Linear(384,2)`는 새로 초기화하고 prior probability 0.01에 맞춘 bias를 사용한다.
5. movable-reference arm에서는 point MLP hidden `.0/.2`만 유지하고 의미가 바뀌는 `point_head.*.4`를 0으로 재초기화한다.

이 방식의 새 모델은 약 44.88M parameters이며, strict-local arm에서는 44,874,294 parameters를 체크포인트로 초기화한다. 신규 파라미터는 class output 770개와 6개 wh output 4,620개, 총 5,390개뿐이다. radial hidden까지 모두 재초기화하는 보수적 방식도 가능하지만 체크포인트 활용률이 약 96.0%로 낮아지므로 기본안으로 삼지 않는다.

`load_state_dict(strict=False)`를 무검증으로 사용하지 않는다. `missing_keys`는 신규 class/wh output으로, `unexpected_keys`는 구 class/radial output으로 정확히 제한하고 그 외 키가 하나라도 있으면 실패시키는 assertion을 둔다. `pe.parametrizations.S.original` 키를 보존하려면 HF 스냅샷의 Cayley parameterization 구조를 그대로 이식해야 한다.

## 4. 탐지 head와 좌표계

### 4.1 출력 계약

각 decoder layer에서 다음을 만든다.

- `pred_logits`: `[B, Q, 2]`, sigmoid VFL용 logits. explicit no-object 출력 없음.
- `pred_center`: 기존 point head가 예측하는 normalized `(cx, cy)`.
- `pred_log_wh`: 신규 wh head의 pixel-space `(log-w, log-h)` residual.
- `pred_boxes`: `[B, Q, 4]` normalized `cxcywh`. `wh_px = exp(pred_log_wh)` 후 `(W,H)`로 나눈다.
- 마지막 layer는 최종 `pred_logits/pred_boxes`, 앞 5개 layer는 `aux_outputs`.

wh는 양수여야 하므로 학습 중 finite/range assertion을 두고, 후처리 단계에서는 이미지 경계로 clip한다. `log(7.5)`를 고정하지 말고 P0에서 train bbox의 축별 중앙값을 산출해 config에 기록한다. 0-init wh output 때문에 학습 시작 시 모든 layer는 이 prior 크기를 내고 이후 log-space residual로 정제된다.

### 4.2 중심점 정책

기존 `relative_to_absolute_pos`는 각 query 중심을 자기 qbs cell 내부로 제한한다. 따라서 한 14.222 px cell에 중심이 두 개 이상 들어가면 구조적으로 둘 다 검출할 수 없다.

- **`strict-local` arm(첫 비교 기준선)**: qbs=256/18, 기존 point head 전체 로드, local sigmoid 좌표계를 유지한다. LSP 정체성과 체크포인트 호환성을 가장 잘 보존한다.
- **`movable-reference` arm(생산 후보)**: query가 적어도 인접 ±1 cell까지 이동하도록 하거나 global inverse-sigmoid reference를 쓴다. point hidden은 로드하고 최종 `.4`는 재초기화한다.
- query를 cell마다 2개로 복제하는 방식은 모델 크기·매처 비용을 키우고 동일 feature로 시작하므로 1차 선택에서 제외한다.

전체 학습 두 번을 바로 돌리지 않는다. P3 소규모 overfit/short validation에서 strict-local과 movable-reference의 matched recall 및 collision cell recall을 비교한 후 production arm 실행 여부를 결정한다.

### 4.3 FeatureSampling 정책

원본은 pixel 좌표를 사실상 normalized grid로 잘못 전달해 초기 query가 학습된 상수에 가깝다. 체크포인트 parity를 위해 첫 arm에서는 이 동작과 `feature_sampling.*` 가중치를 유지한다. 좌표를 올바르게 정규화하는 수정은 입력 token 분포 자체를 바꾸므로 `feature-sampling-fixed` ablation으로 분리한다.

## 5. Dome 프레임워크 연결 계약

### 5.1 모델과 환경

- 신규 registry 모델은 예를 들어 `LSPDetrDetection`으로 만들고 `forward(images, targets=None)`를 지원한다.
- 모델 내부 또는 transform에서 ImageNet normalization을 정확히 한 번만 수행한다.
- `encoder.use_defe=False`를 제공하는 파라미터 없는 stub을 둔다. `det_engine.evaluate()`가 이를 직접 참조한다.
- `src/zoo/__init__.py`에 `from . import lsp`를 추가한다.
- 트렁크 속성명은 `decoder`로 통일해 optimizer regex가 의도대로 동작하게 한다.
- `.deploy()`는 추가하지 않는다. 시작 시 FLOPs 계산 skip 경고는 허용한다.

현재 Dome 환경은 Python 3.11.9, torch 2.13.0+cu130, transformers 5.13.0이다. `hf-5class/modeling.py`의 `transformers.utils.backbone_utils.load_backbone`은 이 환경에서 import되지 않고 `einops`도 설치되어 있지 않다.

- 백본은 network access 없이 `Swinv2Config(image_size=1536, ...)`와 `AutoBackbone.from_config`로 생성한다.
- LSP 소스의 `rearrange`를 유지한다면 `einops==0.8.1`을 Dome 환경 의존성에 명시한다. 무계획한 수동 reshape 치환은 포팅 오류 위험 때문에 피한다.
- Swin stage3/4는 256 학습 때와 다른 window/shift 기하를 사용하므로 late backbone stage를 freeze하지 않고 backbone 저LR로 미세조정한다. patch embedding+stage1 freeze만 Dome 기준선 정책과 맞춘다.

### 5.2 Cayley-STRING 수정

HF 스냅샷의 `CayleySTRING.P`는 `cached_property`이고 eval branch가 이를 사용한다. 첫 validation에서 캐시된 뒤 `S`가 추가 학습되어도 EMA validation이 오래된 `P`를 사용할 수 있다.

- `cached_property`를 제거하고 eval forward마다 `(I-S)(I+S)^-1`를 재계산한다. 32×32 행렬이라 비용은 미미하다.
- train/eval 전환 후 optimizer step을 수행했을 때 eval 결과가 갱신된 `S`를 반영하는 단위 테스트를 추가한다.
- checkpoint key 구조인 `pe.freqs`와 `pe.parametrizations.S.original`은 그대로 유지한다.

### 5.3 criterion과 matcher

stock `DomeCriterion.forward`는 `aux_outputs`가 있으면 `pre_outputs`와 `enc_aux_outputs`도 무조건 읽으므로 LSP 출력에 그대로 사용할 수 없다.

- `LSPCriterion(DomeCriterion)`을 만들고 final+5 aux, 총 6개 output에 대해서만 matching/loss를 계산한다.
- 기존 `loss_labels_vfl`, `loss_boxes`, `_get_go_indices`, `_get_match_pool`은 재사용한다.
- losses는 `vfl`, `boxes`만 사용한다. FGL/DDF `local`은 Dome의 distribution/corner head 전용이라 제외한다.
- Dome parity를 위해 boxes loss는 6개 layer의 union match와 `num_boxes_go` 정규화를 사용한다.
- forward signature는 `(outputs, targets, **kwargs)`로 유지한다.
- empty target과 all-empty batch가 matcher/loss에서 오류 없이 통과하는 테스트를 둔다. 이때 regression L1/GIoU는 0이고, VFL은 모든 query에 대한 finite background-negative loss를 내는 것이 정상이다.

### 5.4 후처리와 평가

`DomePostProcessor`는 `pred_logits`와 normalized `pred_boxes(cxcywh)` 계약이 맞으므로 재사용한다.

- 공식 A/B 지표: `num_top_queries=2000`, class-agnostic NMS IoU 0.7, score 0.01, evaluator maxDets `[500,1000,2000]`.
- 생산 보조 지표: top-k 4000과 이에 맞춘 maxDets 4000 평가를 별도로 산출한다. 공식 2000 결과를 덮어쓰지 않는다.
- query-class flatten top-k 뒤 동일 query가 두 class로 중복될 수 있으므로 class-agnostic NMS 동작과 `all_class_scores` 출력을 smoke test한다.

## 6. 데이터와 전처리

`/home/work/.mnt/combined_all_v1_bundle`는 bbox-only COCO이며 category id가 이미 0/1이라 별도 label remap이 필요 없다.

- train: 18,678 images, 약 4,395만 boxes. 512²/2048²/3375²/4096² 혼재.
- val/test: 전부 1536²이며 bbox 중심이 `[432,1104)` 안에 있어 center-672 평가 규약과 맞는다.
- `MultiCropDatasetWrapper` + `RandomCropWithGrid(1536, center_gt_size=672)`를 유지한다.
- 에폭당 52,802 crops, 총 batch 8 기준 약 6,600 iterations.
- 셀 크기는 소스 전반에서 약 12–16 px로 비슷하므로 별도 scale resize는 하지 않는다.

### 6.1 가짜 dummy GT 제거 — 필수

현재 `RandomCropWithGrid`는 crop 후 box가 0개면 임의 위치에 3×3 box와 label 0을 추가한다. combined에서는 이것이 실제 Non-tumor GT로 학습되어 false positive supervision을 만든다.

- `_add_dummy_box` 호출을 제거하고 shape `[0,4]` boxes와 `[0]` labels를 그대로 유지한다.
- LSP 모델은 denoising을 사용하지 않으므로 target 없는 forward에 문제가 없다.
- Dome 비교군의 denoising 코드도 `max_gt_num==0`을 처리하므로 동일하게 dummy를 제거해 A/B 데이터 의미를 맞춘다.
- empty crop 비율, empty batch 수, class별 GT 수를 첫 epoch 로그에 기록한다.

### 6.2 입력 정규화

체크포인트 processor는 ImageNet mean/std를 사용한다. `/255`만 적용하면 PanNuke에서 학습된 백본·decoder 입력 분포가 깨진다.

- train 순서: crop/색 증강/flip/sanitize → `ConvertPILImage(/255)` → box normalize → GaussianBlur → **ImageNet Normalize**.
- val/test 순서: Resize 1536 → `ConvertPILImage(/255)` → **ImageNet Normalize**.
- Normalize를 model 내부에 넣을지 transform에 넣을지는 한 곳으로 통일하고 double normalization을 assertion/test로 막는다.
- 엄밀한 architecture-only 비교용 from-scratch arm은 별도로 `/255` 조건을 둘 수 있지만 checkpoint fine-tune arm의 기본값은 ImageNet Normalize다.

## 7. 확정 학습 레시피

| 항목 | 값 |
|---|---|
| 하드웨어/배치 | 8×H100, batch 1/GPU, total batch 8, seed 0 |
| 입력 | 1536², qbs=256/18≈14.222, 108²=11,664 queries |
| 에폭 | 30, 약 198K optimizer steps |
| optimizer | AdamW, base lr 2.5e-4, betas [0.9,0.999], wd 1.25e-4 |
| backbone | patch embed+stage1 freeze, 나머지 lr 1.25e-5; backbone norm도 같은 lr/wd 0 |
| decoder/head | lr 2.5e-4; norm만 wd 0(기준 Dome regex와 동일), bias는 기본 wd 적용 |
| schedule | MultiStepLR milestones [24,30], gamma 0.8, warmup 없음 |
| precision | FP32, AMP off, TF32 off |
| EMA/clip | ModelEMA decay 0.9999, warmups 1000, start 0; grad clip 0.1 |
| matcher | focal class/bbox/GIoU cost 3/3/1, matcher alpha 0.25 gamma 2, `use_focal_loss=True` |
| loss | VFL alpha 0.75 gamma 2 ×3 + L1 ×3 + GIoU ×1; final+5 aux union matching |
| postprocess | 공식 top-k 2000 + class-agnostic NMS 0.7/0.01; 보조 top-k 4000 |
| checkpoint | `hf-5class/model.safetensors`, controlled remap/assert load |
| 로깅 | checkpoint every epoch, val/best는 EMA, resolved cfg를 기준 Dome run과 diff |

optimizer regex는 상호 배타적인 세 그룹 이상으로 만든다. HF Swinv2 파라미터 경로에 `.encoder.`가 들어가므로 기존 encoder/decoder norm regex가 backbone LayerNorm을 base LR로 가로채지 않도록 `backbone` 제외 조건을 명시한다. 기준 Dome 설정은 bias를 no-decay로 분리하지 않으므로 LSP도 norm만 wd 0으로 둔다. 런치 전에 그룹별 parameter 이름·개수·LR·WD를 출력하고 전체 parameter가 정확히 한 번만 배정됐는지 assert한다.

## 8. 실행 절차와 게이트

| 단계 | 상태/내용 | 통과 조건 |
|---|---|---|
| P0 체크포인트 감사 | **부분 완료**: 두 포맷 동일성, 432 tensors/45.024M, modern Swinv2 241/241 호환 확인 | 남은 작업: bbox 중앙값 prior 산출, full port controlled-load assert |
| P1 모델 이식 | `src/zoo/lsp/lsp_trunk.py`, `lsp_detr_det.py`, `lsp_criterion.py` | import, CPU construction, checkpoint 예상 키 외 mismatch 0 |
| P2 config/launcher | dataset/include/model config와 `dist_train_lsp.sh` | resolved cfg가 모델 고유 항목·정규화·dummy 제거 외 기준 Dome run과 일치 |
| P3-1 단일 GPU smoke | forward/backward, empty batch, 1536 memory/step profile | loss/grad/boxes finite, output shape 정확, peak <80GB |
| P3-2 checkpoint parity | 256 입력에서 port trunk와 HF snapshot의 embeddings/points 비교 | 허용 오차 내 일치. 의도한 cached-P 수정만 별도 기록 |
| P3-3 overfit | GT<1500인 10장, strict-local과 movable-reference short ablation | AP50≥0.95, matched IoU 상승, collision recall 비교 |
| P3-4 DDP/EMA/val | 2→8 GPU 수십 step + validation 2회 | unused parameter 없음, 두 번째 val이 갱신 weight 반영, evaluator 정상 |
| P4 본 학습 | 선택 arm 30ep, 약 198K steps | 매 epoch checkpoint/eval, NaN 0, optimizer group 불변 |
| P5 Dome 기준선 | 같은 데이터·dummy 제거·공식 top-2000 조건 | LSP와 동일 evaluator/config 차이 대장 확보 |
| P6 test/분석 | 공식 2k + 생산 4k, 그룹별/marker별 결과 | 중복·슬라이드 누수와 query collision을 함께 해석 |

## 9. 구현 체크리스트

- [ ] Swinv2를 `image_size=1536`으로 생성하고 backbone 241/241 strict 호환을 재확인한다.
- [ ] `load_backbone`을 사용하지 않고 `AutoBackbone.from_config`를 사용한다.
- [ ] `einops==0.8.1`을 Dome 환경 의존성에 기록한다.
- [ ] checkpoint remap 후 예상 missing/unexpected key 목록을 assert한다.
- [ ] radial hidden `.0/.2`를 wh hidden으로 이식하고 class/wh output만 새로 초기화한다.
- [ ] wh prior를 train bbox median으로 기록하고 `log(7.5)`를 사용하지 않는다.
- [ ] strict-local/movable-reference의 point output 초기화 차이를 config와 checkpoint 메타에 기록한다.
- [ ] `CayleySTRING.P`의 stale eval cache를 제거하고 train→eval 갱신 테스트를 추가한다.
- [ ] ImageNet Normalize를 train과 val/test에 정확히 한 번 적용한다.
- [ ] `_add_dummy_box` 경로를 제거하고 all-empty batch loss를 테스트한다.
- [ ] `LSPCriterion`은 final+5 aux만 처리하고 FGL/DDF/CDN/DeFE 키를 요구하지 않는다.
- [ ] `encoder.use_defe=False` stub과 registry import를 추가한다.
- [ ] optimizer group이 상호 배타적이고 backbone norm LR이 1.25e-5인지 assert한다.
- [ ] 공식 top-2000 결과와 생산 top-4000 결과를 별도 이름으로 저장한다.
- [ ] 시작 시 FLOPs skip 경고를 정상으로 취급하고 `.deploy()`를 추가하지 않는다.

## 10. 주요 리스크와 완화

| 리스크 | 심각도 | 완화 |
|---|---|---|
| local one-query-per-cell recall ceiling | 높음 | collision metric 추가, strict-local vs movable short ablation, production arm 분리 |
| 가짜 class-0 dummy supervision | 높음 | dummy 제거, empty-target unit/DDP test, 첫 epoch empty 통계 기록 |
| checkpoint 조용한 부분 로드 | 높음 | key remap allowlist와 exact assertion, parameter coverage 출력 |
| Cayley eval의 stale cached `P` | 높음 | cache 제거, validation 2회 사이 weight 반영 단위 테스트 |
| optimizer regex 오배정 | 높음 | 상호 배타 그룹, parameter 단일 배정 및 LR/WD assertion |
| 2000 detection 평가 상한 | 중간 | 공식 2k 유지 + 생산 4k 보조 지표 병기 |
| Swin 256→1536 attention geometry 변화 | 중간 | shape 호환 확인 완료, late stages 저LR fine-tune, freeze 금지 |
| matcher cost matrix/FP32 memory | 중간 | GPU cost 구축 순차화 또는 GIoU chunk, P3에서 peak 측정 |
| compile×DDP 및 환경 API | 중간 | static compile smoke, modern AutoBackbone, pinned einops |
| PanNuke→IHC domain gap | 낮음 | ImageNet normalization 유지, 전체 trunk fine-tune, from-scratch 비교 선택 |

## 11. 참고 파일

- 마스터 플랜: `/home/work/tksong/AIVIS-DETECTION/docs/260817-lsp-detr-port-plan.md`
- 체크포인트: `/home/work/tksong/lsp-detr/hf-5class/{config.json,model.safetensors,model_5class.pth,modeling.py,preprocessor_config.json}`
- LSP 원본: `/home/work/tksong/lsp-detr/lsp_detr/modeling/{lsp_detr.py,criterion.py,matcher.py}`
- Dome 모델/손실: `/home/work/tksong/AIVIS-DETECTION/AIVIS-Dome-DETR/src/zoo/dome/{dome.py,dome_criterion.py,matcher.py,postprocessor.py}`
- Dome 엔진/데이터: `/home/work/tksong/AIVIS-DETECTION/AIVIS-Dome-DETR/src/solver/det_engine.py`, `src/data/transforms/_transforms.py`, `src/data/dataset/coco_dataset.py`
- 데이터셋: `/home/work/.mnt/combined_all_v1_bundle/{train_coco.json,val_coco.json,test_coco.json}`
