# P4 strict-local 학습 종료 후 작업 목록

작성 2026-08-19. 대상 런: `[combined]LSP-T_strict-local_H100x8_hf5class-ft_30ep`
(`/home/work/.mnt/DET_RESULT/lsp_detr/`, 완료 예상 ~08-26). **학습이 돌아가는 동안에는 아무것도 적용하지 않는다** —
아래 코드 수정은 전부 import 시점/리줌 경로에 닿으므로 런 종료(또는 확정 중단) 후에만 작업한다.
근거: 외부 리뷰 5건 검토 결과(2026-08-19, 본 문서 §1–§5)와 자체 발견(§6–§8). 검토 상세는 대화 기록과
IMPLEMENTATION_LOG.md 참조. 우선순위: **A = 다음 런(P5/movable) 전에 필수, B = 체크포인트 배포 전에 필수, C = 정리**.

---

## §1 [A] Dome 의존성 고정 — 커밋 안 된 `_get_match_pool` 패치 (리뷰 item 1, MAJOR)

**사실**: `lsp_det/lsp_criterion.py:16`이 import하는 `_get_match_pool`은
`AIVIS-DETECTION/AIVIS-Dome-DETR/src/zoo/dome/dome_criterion.py`의 **커밋되지 않은 working-tree 패치**에만 존재한다
(레포는 단일 커밋 1e6a557). 깨끗한 checkout에서는 `import lsp_det` 자체가 ImportError로 죽어
train/inference/eval_topk/pytest 전부 실행 불가(외부 리뷰어 환경에서 재현된 증상, 로컬에서 git archive로 확인).
패치는 결과 동일(순서 보존 스레드풀)이므로 본 런의 정확성과는 무관 — 재현성 문제다.

- [x] (2026-09-15) `lsp_criterion.py`의 import를 fallback으로 변경:
  ```python
  try:
      from src.zoo.dome.dome_criterion import _get_match_pool
  except ImportError:          # pristine Dome@1e6a557: sequential matching (result-identical, slower)
      def _get_match_pool():
          return None
  ```
- [x] (2026-09-15, 더 강하게 해결) Dome 런타임 전체를 `det/third_party/dome/`에 벤더링(패치 적용된 working tree 그대로, VENDORED.md) → 클린 클론에서 sibling 레포 없이 import/학습 가능. ~~패치를 diff로 동봉:~~ `git -C /home/work/tksong/AIVIS-DETECTION diff HEAD -- AIVIS-Dome-DETR/src/zoo/dome/dome_criterion.py > det/patches/dome_criterion_matchpool.diff` (선택: AIVIS-DETECTION에 커밋하는 쪽이 더 깔끔)
- [x] (2026-09-15) README.md / `lsp_det/__init__.py`의 "Dome은 수정하지 않음" 문구 정정 → "det/가 수정하지 않음; Dome@1e6a557 필요, 매처 스레드 패치는 선택(성능만)"
- [x] (2026-09-15) VENDORED.md에 Dome 커밋 해시(1e6a557) + 패치 내용 명시; `DOME_MATCH_THREADS`는 벤더링 본에서 항상 유효
- [ ] (선택) `bootstrap_dome()`이 Dome의 git HEAD/dirty 상태를 `init_report.json`에 기록

## §2 [A] resume arm 가드 (리뷰 item 3, MAJOR — movable 런 전 필수)

**사실**: 두 arm은 state_dict 키·shape가 완전히 동일하고(`center_span`은 float 속성, `log_wh_prior`는
persistent=False 버퍼, `lsp_trunk.py:330`), Dome resume은 키/shape만 검사하므로 **잘못된 arm/wh_prior로
리줌해도 에러가 없다**. 게다가 `train.py:59-76`이 solver 생성 전에 `init_report.json`을 현재 설정으로
덮어쓰고, 런처는 `ARM` 기본값(strict-local)을 `RESUME`과 무관하게 항상 `-u`에 붙이며 기본 OUTPUT_DIR에
`${ARM}`이 들어가 있어 — movable 런을 `ARM=` 없이 리줌하면 strict-local로 바뀐 채 **기존 strict-local 런
디렉터리에 로그/체크포인트를 섞어 쓴다**(tee -a).

- [ ] `train.py`: `args.resume` 지정 시 기존 `init_report.json`(output_dir, 없으면 dirname(resume))을 먼저 읽어
  `center_mode` / `center_span_cells` / `wh_prior_px` 일치 assert; 불일치 시 두 값을 출력하고 중단;
  일치 시 리포트를 덮어쓰지 말고 `resumed_from`/`resume_dates` 필드만 append
- [ ] `dist_train_lsp.sh`: `RESUME`이 설정되고 `ARM`이 명시되지 않았으면 리줌 디렉터리의 init_report.json에서
  ARM을 유도하거나 에러로 거부; OUTPUT_DIR 불일치도 동일하게 검사
- [ ] (선택, 키 셋이 바뀌므로 반드시 런 종료 후) `log_wh_prior`를 persistent로 바꾸고 1-element `center_span`
  버퍼 추가 → 로드 후 값 비교로 하드 가드. 기존 체크포인트(432키)는 legacy 경로로 tolerant load 필요

## §3 [B] 체크포인트 자기서술화 / 배포 규칙 (리뷰 item 4, MINOR→배포시 MAJOR)

**사실**: .pth에는 arm 메타가 없고 모델에도 arm을 구분할 persistent 텐서가 없다. `inference.py:251-288`이
사이드카 `init_report.json`으로 arm을 검증하지만, **.pth 단독 복사본**은 경고 후 yml 기본(strict-local)으로
fallback — movable 체크포인트면 조용히 틀린 박스가 나온다.

- [ ] 당장(코드 불요): 체크포인트를 외부로 전달할 때 **반드시 `init_report.json`을 같은 디렉터리에 동봉**하는
  규칙을 README에 명문화; 리뷰어에게도 현 런의 init_report.json 전달
- [ ] `inference.py`: 리포트가 없을 때 warn+fallback 대신 `--center-mode` 명시를 요구(에러)로 강화
- [ ] (선택) §2의 persistent buffer가 들어가면 checkpoint 자체 검증으로 대체 가능

## §4 [C] 절대 경로 정리 (리뷰 item 2, MINOR)

전부 `-u`/env로 오버라이드 가능하고 Dome 기준선도 같은 관행이므로 낮은 우선순위. 이식 필요가 생기면:

- [x] (2026-09-15) `lsp_swinv2.yml`의 `pretrained:`를 레포 상대 경로(`hf-5class/model.safetensors`, `LSPDetrDetection.resolve_pretrained_path`)로 변경
  (또는 README에 `-u LSPDetrDetection.pretrained=...` 오버라이드 예시 추가)
- [x] (2026-09-15, 일부) README "Setup on a new machine" + `scripts/fetch_hf5class.py`(HF hub 리비전 a32176184e 고정, sha256 검증) + `det/requirements.txt`; 남은 것: ~~README에 재현 섹션 추가: hf-5class 가중치 출처(HF hub `RationAI/LSP-DETR`) + 다운로드 한 줄 + sha256,~~
  `train_coco_areafix.json` 재생성법(`scripts/fix_train_area.py`), conda env export(dome env, einops 수동 설치 포함)
- [ ] `scripts/eval_topk.py:36`의 test json 하드코딩을 인자로 오버라이드 가능하게

## §5 [C] non-finite loss 관측성 (리뷰 item 5, MINOR)

**사실**: `lsp_criterion.py:107`의 `nan_to_num(nan=0.0)`은 Dome 패리티(dome_criterion.py:564 동일).
Inf는 1e38로 로그에 보이므로 숨지 않고, 숨는 건 NaN→0.0. 실제 최악 경로는 non-AMP에서 NaN grad →
`clip_grad_norm_`이 전 파라미터를 NaN으로 → 이후 모든 loss가 0.0(유한)이라 엔진이 중단하지 않고
죽은 모델로 완주하는 것. 현 런 로그에는 NaN/Inf 없음(주기적으로 확인).

- [ ] criterion에 non-finite 항 카운터 추가(rank 태그 print + `Loss/nonfinite_count` TB scalar) —
  `nan_to_num` 자체는 유지(수치 패리티)
- [ ] 로그 감시 규칙 문서화: Loss/* 항이 정확히 0.0(NaN 흔적) 또는 >1e30(Inf 흔적), 혹은 total 0.0이면
  죽은 런 → 직전 체크포인트에서 재시작
- [ ] 엔진의 `clip_grad_norm_(error_if_nonfinite=True)` 전환은 Dome 패리티를 깨므로 **하지 않는다**(기록만)

## §6 [A] 본 런 평가·분석 (FINETUNE_STRATEGY §9, P6 준비)

- [ ] best epoch 확정 후 `scripts/eval_topk.py`로 보조 top-4000 지표 실행(val; 아직 한 번도 학습된
  체크포인트에 실행된 적 없음 — open item 5)
- [ ] 클래스별(Non-tumor/Tumor)·area별 AP/AR 표 정리, collision cell recall 분석(strict-local 이론 한계 대비)
- [ ] epoch 0 데이터 보정: 리줌으로 `log.txt`에 epoch 0 행이 없고 best_stat도 epoch 0(AP 0.429)을 모른다.
  곡선/best 판단 시 `launcher_launch2.log`의 ep0 수치(AP 0.429/AP50 0.804)를 수동 포함;
  ep0 상태는 `checkpoint0000.pth`에 있음
- [ ] `train_run.log`는 launch #2+#3 concat(tee -a)이므로 파싱 시 두 번째 `cfg:` 라인에서 분리
- [ ] 결과를 HYPERPARAMS.md §9와 IMPLEMENTATION_LOG.md에 최종 기록

## §7 [A] 다음 런 준비 (movable-reference, P5 Dome-M 기준선)

- [ ] §2 resume 가드 먼저 머지 → `ARM=movable-reference bash det/scripts/dist_train_lsp.sh` (출력 디렉터리는
  ARM이 이름에 들어가므로 자동 분리; NFS `OUTPUT_ROOT` 기본값 확인)
- [ ] P5 Dome-M 2-class 기준선 런은 **같은 파생 train json**(`train_coco_areafix.json`)과 dummy 제거·top-2000
  조건을 사용해야 함(HYPERPARAMS.md §8 대장 참조)
- [x] ~~(속도, 선택) GPU Hungarian(torch-linear-assignment) 검증~~ → **2026-08-20 완료, 결론: 도입 금지(취소)**.
  Phase-1 벤치(det/.gpu_lap/results/REPORT.md, identity_bench.json) 결과: 결과 동일성은 완벽(실측 11개 행렬 전부
  scipy와 할당·총비용 f64 delta 0.0, 동점 스트레스 포함)이지만, CUDA auction 커널이 **실제 매처 비용행렬에서
  scipy 대비 GT=500에서 35배, GT≥1239에서 ~130배 느림**(quiet GPU 실측 304s vs 2.3s @11664×1239; ~T^4-5 스케일링).
  6레이어 배치 1콜도 단건 대비 ~1.1배뿐. 랜덤 행렬 벤치는 실데이터 대비 solver 난이도를 10~66배 과소평가하므로
  향후 어떤 대안 solver든 **det/.gpu_lap/matrices/의 실제 행렬로 벤치할 것**. 부수 발견: scipy 자체도 실데이터에서
  ~T^2.5로 열화(2.3s@1239 → 16.9s@2739) → 진짜 레버는 solver 교체가 아니라 **Hungarian 진입 전 후보 축소**
  (GT별 top-k 쿼리 프루닝, aux 레이어 근사 매칭+final만 정확 매칭 등). 재도전 시 주의: 미실행 케이스(11664×2739
  전체, W<T 방향 실데이터)의 동일성은 미확인. 패키지 빌드 방법(CUDA13 pip 툴체인 함정 포함)은
  .gpu_lap/results/install_commands.txt에 보존.
- [ ] movable 런 전 `feature_sampling_fixed`(현재 False = 스냅샷 픽셀좌표 grid_sample 패리티) 유지/수정 재결정

## §8 [C] 기타 정리

- [ ] dead key 정리: `combined_tnt_detection.yml`의 `num_workers 8`(실제 4/2로 Dome include가 덮음),
  `LSP-T-combined.yml`의 `visualization_epoch_interval 5`(solver는 기본 1 사용) — 주석 정정 또는 실효화
- [ ] `det/HYPERPARAMS.md`, `det/README.md` 변경분, 본 문서를 lsp-detr 레포에 커밋
- [ ] 홈 디스크 재점검(현재 36/49GB): 필요 시 `~/.cache/pip` 3.1GB 정리

## §9 [B] HER2 런 관련 (2026-09-04 리뷰)
- [ ] `scripts/eval_topk.py --split test`가 TNT bundle test_coco.json을 하드코딩(:36) + img_folder 미교체 → HER2 런에는 --test-ann/--test-img-folder 인자화 필요
