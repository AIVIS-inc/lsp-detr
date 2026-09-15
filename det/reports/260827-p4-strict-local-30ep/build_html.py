#!/usr/bin/env python
"""Build the shareable HTML training record (figures inlined as data URIs, epoch table from CSV)."""
import base64, csv, json, os, html
R = "/home/work/tksong/lsp-detr/det/reports/260827-p4-strict-local-30ep"
OUT = "/tmp/claude-1100/-home-work-tksong/0b6c5eff-f0a8-4066-9213-1854a681594a/scratchpad/lsp-t-strict-local-run.html"

def img(name):
    b = open(os.path.join(R, name), "rb").read()
    return "data:image/png;base64," + base64.b64encode(b).decode()

rows = list(csv.DictReader(open(os.path.join(R, "epoch_metrics.csv"))))
S = json.load(open(os.path.join(R, "summary.json")))

def f4(x): return f"{float(x):.4f}"
tr = []
for i, r in enumerate(rows):
    e = int(r["epoch"]); prev = rows[i - 1] if i else None
    dap = "" if not prev else f"{(float(r['AP']) - float(prev['AP'])) * 100:+.2f}"
    cls = ' class="lrdrop"' if e == 24 else (' class="best"' if e == 29 else "")
    def opt(k, fmt): return "" if r[k] == "" else format(float(r[k]), fmt)
    tr.append(f"<tr{cls}><td>{e}</td><td>{float(r['lr']):.2e}</td><td>{float(r['train_loss']):.3f}</td>"
              f"<td>{opt('train_loss_vfl', '.3f')}</td><td>{opt('train_loss_giou', '.3f')}</td><td>{opt('train_loss_bbox', '.4f')}</td>"
              f"<td>{f4(r['AP'])}</td><td>{f4(r['AP50'])}</td><td>{f4(r['AR@2000'])}</td><td>{dap}</td>"
              f"<td>{float(r['train_s_per_it']):.2f}</td><td>{int(r['train_time_s'])/3600:.2f}</td><td>{int(r['eval_time_s'])//60}</td><td>{r['checkpoint_mtime'][5:]}</td></tr>")
table = "\n".join(tr)

page = f"""<title>LSP-T strict-local 학습 기록</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  --ground:#f6f7f5; --panel:#ffffff; --ink:#1a1e24; --ink-2:#4d5661; --ink-3:#7a8490; --rule:#d7dce2; --rule-soft:#e8ebee;
  --accent:#2a78d6; --accent-ink:#1c5cab; --accent-wash:#e6effb; --warm:#eb6834; --warm-wash:#fdeee6; --ok:#0f8a3d;
  --shadow:0 1px 2px rgba(20,30,40,.06);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground:#14171b; --panel:#1c2026; --ink:#e9ecef; --ink-2:#aeb6bf; --ink-3:#7f8893; --rule:#2f363e; --rule-soft:#262c33;
    --accent:#3987e5; --accent-ink:#86b6ef; --accent-wash:#1b2b40; --warm:#f0784a; --warm-wash:#3a2418; --ok:#3fbf6f;
    --shadow:none;
  }}
}}
:root[data-theme="dark"] {{
  --ground:#14171b; --panel:#1c2026; --ink:#e9ecef; --ink-2:#aeb6bf; --ink-3:#7f8893; --rule:#2f363e; --rule-soft:#262c33;
  --accent:#3987e5; --accent-ink:#86b6ef; --accent-wash:#1b2b40; --warm:#f0784a; --warm-wash:#3a2418; --ok:#3fbf6f;
  --shadow:none;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--ground); color:var(--ink); font-family:"IBM Plex Sans","Noto Sans KR","Apple SD Gothic Neo",system-ui,sans-serif; font-size:15.5px; line-height:1.65; -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:1080px; margin:0 auto; padding:40px 28px 80px; }}
h1,h2,h3 {{ font-family:"IBM Plex Sans Condensed","IBM Plex Sans",system-ui,sans-serif; text-wrap:balance; margin:0; letter-spacing:-.005em; }}
h1 {{ font-size:2.35rem; font-weight:600; line-height:1.12; }}
h2 {{ font-size:1.45rem; font-weight:600; margin-top:56px; padding-top:18px; border-top:1px solid var(--rule); }}
h3 {{ font-size:1.05rem; font-weight:600; margin-top:28px; color:var(--ink); }}
p {{ max-width:70ch; margin:12px 0; }}
.eyebrow {{ font-family:"IBM Plex Mono",monospace; font-size:.74rem; letter-spacing:.08em; text-transform:uppercase; color:var(--ink-3); margin-bottom:12px; }}
.lede {{ color:var(--ink-2); font-size:1.05rem; max-width:64ch; margin-top:14px; }}
code, .mono {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.88em; }}
code {{ background:var(--rule-soft); padding:1px 5px; border-radius:3px; }}
pre {{ background:var(--panel); border:1px solid var(--rule); padding:14px 16px; overflow-x:auto; font-family:"IBM Plex Mono",monospace; font-size:.85rem; line-height:1.5; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:28px 0 8px; }}
.tile {{ background:var(--panel); border:1px solid var(--rule); padding:14px 16px 12px; box-shadow:var(--shadow); }}
.tile .k {{ font-family:"IBM Plex Mono",monospace; font-size:.72rem; letter-spacing:.06em; text-transform:uppercase; color:var(--ink-3); }}
.tile .v {{ font-family:"IBM Plex Sans Condensed",sans-serif; font-size:2rem; font-weight:600; line-height:1.1; margin-top:4px; font-variant-numeric:tabular-nums; }}
.tile .d {{ font-size:.82rem; color:var(--ink-2); margin-top:4px; font-variant-numeric:tabular-nums; }}
.tile .d b {{ color:var(--accent-ink); font-weight:600; }}
.verdict {{ border-left:3px solid var(--accent); background:var(--accent-wash); padding:14px 18px; margin:24px 0; max-width:none; }}
.verdict p {{ margin:6px 0; max-width:none; }}
.caution {{ border-left:3px solid var(--warm); background:var(--warm-wash); padding:12px 18px; margin:18px 0; }}
.caution p {{ margin:6px 0; max-width:none; }}
figure {{ margin:22px 0; background:var(--panel); border:1px solid var(--rule); padding:10px 10px 6px; box-shadow:var(--shadow); }}
figure img {{ display:block; width:100%; height:auto; max-width:100%; }}
figcaption {{ font-size:.86rem; color:var(--ink-2); padding:8px 6px 4px; line-height:1.5; }}
.figgrid {{ display:grid; grid-template-columns:1fr; gap:4px; }}
.tablewrap {{ overflow-x:auto; border:1px solid var(--rule); background:var(--panel); box-shadow:var(--shadow); }}
table {{ border-collapse:collapse; width:100%; font-size:.86rem; font-variant-numeric:tabular-nums; }}
th,td {{ padding:6px 10px; text-align:right; border-bottom:1px solid var(--rule-soft); white-space:nowrap; }}
th {{ font-family:"IBM Plex Mono",monospace; font-weight:500; font-size:.72rem; letter-spacing:.04em; text-transform:uppercase; color:var(--ink-3); background:var(--ground); position:sticky; top:0; }}
td:first-child, th:first-child {{ text-align:left; padding-left:14px; }}
.kv td:first-child {{ color:var(--ink-2); width:160px; white-space:normal; }}
.kv td {{ text-align:left; white-space:normal; }}
tr.lrdrop td {{ border-top:2px solid var(--accent); }}
tr.best td {{ font-weight:600; background:var(--accent-wash); }}
.tl {{ display:grid; grid-template-columns:150px 1fr; gap:0 18px; max-width:none; }}
.tl div {{ padding:9px 0; border-bottom:1px solid var(--rule-soft); }}
.tl .t {{ font-family:"IBM Plex Mono",monospace; font-size:.8rem; color:var(--ink-2); padding-top:11px; }}
.tl .bad {{ color:var(--warm); font-weight:600; }}
ul {{ padding-left:20px; max-width:74ch; }} li {{ margin:6px 0; }}
.small {{ font-size:.84rem; color:var(--ink-2); }}
@media (max-width:640px) {{ .tl {{ grid-template-columns:1fr; }} .tl .t {{ padding-bottom:0; border:0; }} h1 {{ font-size:1.9rem; }} }}
</style>
<div class="wrap">
<div class="eyebrow">LSP-DETR-T bbox port · P4 · combined_all_v1_bundle · 2026-08-18 → 08-27</div>
<h1>strict-local 30 epoch 학습 기록</h1>
<p class="lede">런 <span class="mono">[combined]LSP-T_strict-local_H100x8_hf5class-ft_30ep</span>의 전 구간 기록. 모든 수치는 런 디렉터리의 <span class="mono">log.txt</span>·<span class="mono">train_run.log</span>·체크포인트 mtime에서 파싱했고, 두 로그의 COCO 출력이 ep1–29 전 항목에서 일치함을 확인했다.</p>

<div class="tiles">
  <div class="tile"><div class="k">AP@[.5:.95]</div><div class="v">{S['ep29']['AP']:.4f}</div><div class="d">ep0 {S['ep0']['AP']:.3f} → <b>+{(S['ep29']['AP']-S['ep0']['AP']):.3f}</b></div></div>
  <div class="tile"><div class="k">AP50</div><div class="v">{S['ep29']['AP50']:.4f}</div><div class="d">ep0 {S['ep0']['AP50']:.3f} → <b>+{(S['ep29']['AP50']-S['ep0']['AP50']):.3f}</b></div></div>
  <div class="tile"><div class="k">AR@2000</div><div class="v">{S['ep29']['AR']:.4f}</div><div class="d">ep0 {S['ep0']['AR']:.3f} → <b>+{(S['ep29']['AR']-S['ep0']['AR']):.3f}</b></div></div>
  <div class="tile"><div class="k">best epoch</div><div class="v">29 / 29</div><div class="d">val AP 30 epoch 연속 상승</div></div>
  <div class="tile"><div class="k">wall time</div><div class="v">8d 20h</div><div class="d">7.2–7.5 h / epoch · 8 × H100</div></div>
</div>

<div class="verdict">
<p><strong>결론.</strong> 과적합 없이 30 epoch에 거의 정확히 수렴했다. 마지막 3 epoch의 ΔAP는 +0.08 / +0.03 / +0.10 pp, AP50은 ep25 이후 합계 +0.06 pp — 같은 LR로 더 돌릴 가치는 낮다. best = 최종 = <span class="mono">best_stg1.pth</span>(ep29 EMA).</p>
<p>같은 데이터셋의 Dome 기준선(P5)은 아직 없어 “LSP가 Dome보다 나은가”는 이 런만으로 판정할 수 없다.</p>
</div>

<h2>결과</h2>
<figure><img src="{img('fig3_val_metrics.png')}" alt="Validation AP50, AR@2000, AP per epoch"><figcaption>검증 지표(EMA 가중치, val 9,853장, top-2000 + class-agnostic NMS 0.7). AP는 매 epoch 상승, AP50은 ep14부터 0.870–0.871에 머물다 ep24 LR ×0.8 이후에만 움직였다.</figcaption></figure>
<figure><img src="{img('fig4_epoch_deltas.png')}" alt="Per-epoch delta AP and delta AP50"><figcaption>epoch별 이득(pp). AP는 ep1–5 +1.8–2.0 → ep10 +0.5 → ep20 이후 +0.1 수준. AP50은 ep14 이후 ep25(+0.34, LR drop 직후)를 빼면 ±0.05.</figcaption></figure>

<h3>학습 손실</h3>
<figure><img src="{img('fig1_train_loss.png')}" alt="Train loss per iteration and epoch"><figcaption>총 train loss(final + aux 5층). iteration 값은 100 it마다 기록된 20-step 중앙값. 30 epoch 내내 단조 하강, ep24 drop에서 −0.085(평소 −0.03의 3배).</figcaption></figure>
<figure><img src="{img('fig2_loss_components.png')}" alt="VFL, GIoU, L1 loss components"><figcaption>손실 항별(epoch 평균). 세 항 모두 끝까지 감소. L1은 ep8 이후 0.027–0.029 바닥 근처. aux layer 0은 final layer를 일정한 간격으로 따라간다.</figcaption></figure>

<h3>epoch 테이블</h3>
<p class="small">lr = 백본 그룹(log.txt <span class="mono">train_lr</span>). loss는 epoch 평균(ep0은 launch #2 iteration 로그의 러닝 평균). s/it·train h는 학습 구간, eval은 검증 소요, ckpt는 <span class="mono">checkpointNNNN.pth</span> mtime. 파란 선 = LR drop 적용 첫 epoch, 강조 행 = best. 전체 13개 COCO stat은 <span class="mono">epoch_metrics.csv</span>.</p>
<div class="tablewrap"><table>
<thead><tr><th>ep</th><th>lr</th><th>loss</th><th>VFL</th><th>GIoU</th><th>L1</th><th>AP</th><th>AP50</th><th>AR@2000</th><th>ΔAP pp</th><th>s/it</th><th>train h</th><th>eval min</th><th>ckpt (KST)</th></tr></thead>
<tbody>
{table}
</tbody></table></div>

<h2>학습 추이 해석</h2>
<ul>
<li><strong>세 단계.</strong> ep0–8 급상승(AP +1.1–2.0 pp/epoch, 새 class/wh head가 자리잡는 구간) → ep9–23 완만(+0.1–0.7 pp, AP50은 ep14부터 정체) → ep24–29 LR drop 이후(AP50이 10 epoch 만에 처음 움직여 0.8743–0.8749).</li>
<li><strong>AP는 오르는데 AP50은 멈춘 이유.</strong> 후반 이득은 IoU 0.5 탐지가 아니라 더 높은 IoU 문턱의 박스 정합에서 나왔다. 14 px 안팎의 핵 박스라 IoU ≥ 0.75는 본질적으로 어렵고, 실무 지표로는 AP50이 더 적절하다.</li>
<li><strong>train loss는 끝까지 직선 하강(−0.023/epoch)</strong>하지만 val 이득은 0에 수렴 — train–val 갭이 벌어지기 시작하는 단계. 아직 과적합은 아니다.</li>
<li><strong>LR 감소에만 반응했다.</strong> 10 epoch 동안 유일한 AP50 움직임이 drop 직후였다. 더 짜낸다면 epoch 연장이 아니라 LR annealing(5 epoch, 1e-5 → 1e-6)이며 기대 이득 AP +0.2–0.5 pp / AP50 +0.1–0.3 pp, 비용 ~1.5일. 포화 곡선 외삽: ep24–29 피팅 점근 AP 0.616, ep15–29 피팅 0.632(낙관). <strong>추천: 여기서 마감</strong>, GPU는 P5 Dome 기준선 / movable-reference 아암에.</li>
<li>30 epoch·milestone [24, 30] 예산은 이 세팅에 거의 정확히 맞았다(ep29 = best, 낭비 epoch 없음).</li>
</ul>

<h2>런 정의</h2>
<div class="tablewrap"><table class="kv"><tbody>
<tr><td>모델</td><td><span class="mono">LSPDetrDetection</span> — LSP-DETR-T(Swinv2-T, HF <span class="mono">hf-5class</span> 스냅샷) 핵 분할 → 박스 탐지 포트 (<span class="mono">det/lsp_det/</span>)</td></tr>
<tr><td>아암</td><td><strong>strict-local</strong> — <span class="mono">center_span_cells=1.0</span>, <span class="mono">wh_prior_px=[14,14]</span></td></tr>
<tr><td>초기화</td><td>hf-5class 432 텐서 중 418 로드(99.99 %), 신규 5,390 파라미터 = class_head(bias prior 0.01) + wh_head 최종층(zero); 백본 45 텐서 동결(embeddings + encoder.layers.0)</td></tr>
<tr><td>데이터</td><td><span class="mono">combined_all_v1_bundle</span> 2-class(Tumor / Non-tumor). train = <span class="mono">combined_all_v1_bundle_derived/train_coco_areafix.json</span>(ki67_NET 192만 ann에 area/iscrowd 보강), val 9,853장 — GT는 중앙 672² 안에만 존재</td></tr>
<tr><td>입력</td><td>학습 <span class="mono">RandomCropWithGridNoDummy</span> 1536², centre_gt 672 + ColorJitter / Flip / Blur + ImageNetNormalize; 평가 Resize 1536²</td></tr>
<tr><td>배치·하드웨어</td><td>8 × H100 80GB, GPU당 1장(total 8), <span class="mono">OMP_NUM_THREADS=1</span>, <span class="mono">DOME_MATCH_THREADS=6</span></td></tr>
<tr><td>옵티마이저</td><td>AdamW lr 2.5e-4 / wd 1.25e-4; 백본 lr 1.25e-5(norm은 wd 0); betas (0.9, 0.999); warmup 0</td></tr>
<tr><td>스케줄</td><td>MultiStepLR milestones [24, 30], γ 0.8 → ep24부터 ×0.8</td></tr>
<tr><td>손실</td><td>LSPCriterion: VFL + L1 + GIoU, final + aux 5층 (denoising·DeFE 없음)</td></tr>
<tr><td>평가</td><td>EMA 가중치, Dome AITOD COCO 평가기(top-2000, class-agnostic NMS 0.7, score 0.01), 8-GPU 분산 평가; best 선택 기준 = AP</td></tr>
<tr><td>정밀도·환경</td><td>TF32 on(<span class="mono">--tf32 keep</span>, 머신 전역 <span class="mono">TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1</span>) · torch 2.13.0+cu130 · NCCL 2.29.7 · transformers 5.13.0 · seed 0</td></tr>
<tr><td>명령</td><td><span class="mono">RESUME=&lt;run&gt;/last.pth setsid nohup bash det/scripts/dist_train_lsp.sh</span> → <span class="mono">torch.distributed.run --nproc_per_node=8 train.py -c configs/LSP-T-combined.yml --seed 0 --tf32 keep -r &lt;last.pth&gt; -u epoches=30 … center_mode=strict-local</span></td></tr>
</tbody></table></div>

<h2>타임라인·사건</h2>
<div class="tl">
<div class="t">08-18 오전</div><div><span class="bad">launch #1</span> 시작 직후 <span class="mono">KeyError 'area'</span>(ki67_NET train ann에 area/iscrowd 없음) → <span class="mono">train_coco_areafix.json</span> 파생, config 교체</div>
<div class="t">08-18 ~10:13–15:25</div><div><strong>launch #2</strong> ep0 학습 5:11:44(2.83 s/it) + eval → AP 0.429 / AP50 0.804, <span class="mono">checkpoint0000.pth</span></div>
<div class="t">08-18 15:43</div><div><span class="bad">launch #2 ENOSPC 사망</span> — 출력이 49 GB 홈 loop 디스크에 있었음. rank0 <span class="mono">torch.save</span> iostream error, rank4/6 <span class="mono">train_samples</span> makedirs 실패(<span class="mono">train_run.log:318-356</span>). best_stg1 손상, last/checkpoint0000 정상</div>
<div class="t">08-18 16:14</div><div><strong>launch #3 = resume</strong> — 출력을 <span class="mono">/home/work/.mnt/DET_RESULT/lsp_detr/</span>로 옮기고 ep0 <span class="mono">last.pth</span>에서 재개, <span class="mono">setsid nohup</span> 분리 실행</div>
<div class="t">08-19 12:15, 19:04 · 08-20 23:13</div><div>CUDA caching-allocator OOM <em>경고</em> 15건(7.7–8.9 GB 블록 할당 실패 → 캐시 해제 후 재시도 성공). 이후 재발 없음. rank5는 런 내내 ~78 GB 예약 유지(캐시)</div>
<div class="t">08-24</div><div>VSCode 창 종료 — 런은 PPID 1·자체 세션이라 무영향(13:45 점검)</div>
<div class="t">08-25 22:05</div><div>ep24 체크포인트 — LR drop이 적용된 첫 epoch</div>
<div class="t">08-27 12:19 / 12:34</div><div><span class="mono">checkpoint0029.pth</span> / 최종 eval → <span class="mono">best_stg1.pth</span>(ep29). solver 보고 <span class="mono">Training time 8 days, 20:17:09</span>(launch #3, ep1–29). GPU 8장 유휴</div>
</div>

<h3>시간·자원</h3>
<figure><img src="{img('fig5_epoch_time.png')}" alt="Seconds per iteration per epoch"><figcaption>스텝 시간 2.83 → 4.07 s/it. 원인은 GPU가 아니라 CPU Hungarian matcher(11,664 query × GT 수천): 8 GPU 사용률이 0–100 %를 주기적으로 오가는 rank 간 대기 패턴. 외부 경합 없음(load ≈ 21 / 64 core, 전부 학습 자체).</figcaption></figure>
<p>rank0 <span class="mono">max mem</span> 27.1 GB, 실 예약 30–38 GB(rank5 78 GB). 체크포인트 30 × 716 MB + best + last ≈ <strong>22.9 GB</strong>(NFS, 17 TB 여유).</p>

<h2>수치 인용 전 주의</h2>
<div class="caution">
<p><strong>이 평가기의 13개 stat에 AP75는 없다.</strong> 순서: AP, AP50, AP_verytiny, AP_tiny, AP_small, AP_medium, AR@500/1000/2000, AR_vt/tiny/small/medium. 학습 중 점검(08-25~27)에서 index 2(ep29 0.4553)를 “AP75”로 부른 것은 오기 — 실제로는 AP_verytiny다.</p>
<p><strong>area-subset 통계(verytiny/tiny/small/medium)는 8-GPU 평가에서 오염됨.</strong> DistributedSampler 패딩(9,853 → 9,856) + <span class="mono">coco_eval_aitod.merge()</span>가 eval_imgs를 dedup하지 않아 (class, area) 블록이 어긋난다. 전체 AP/AP50/AR@2000은 ≤ 0.001 차이로 정확. 정확한 subset 값은 <span class="mono">det/inference.py TILING=off</span>(단일 프로세스) 재평가로.</p>
</div>
<ul>
<li>val GT가 중앙 672²에만 있으므로 모델은 1536 입력 중앙에서만 발화 → 실제 추론은 <span class="mono">det/inference.py TILING=auto</span>(stride 672 타일링) 필수. 이미지 리사이즈 금지.</li>
<li>best 선택 기준은 AP(<span class="mono">coco_eval_bbox[0]</span>). test_eval 킷의 다른 대상은 AP50 선택인 경우가 있어 혼동 주의.</li>
<li>TF32가 켜진 채 학습·평가(Dome 런들과 동일). 순수 FP32 재현은 <span class="mono">--tf32 off</span>.</li>
<li><span class="mono">log.txt</span>에는 ep0 행이 없다(launch #2 ENOSPC). ep0 수치는 <span class="mono">train_run.log</span>의 COCO 출력에서 복원.</li>
<li>체크포인트에는 아암 정보가 없다 — <span class="mono">init_report.json</span>이 유일한 기록. resume·inference 시 반드시 함께 보관(POST_TRAINING_TODO §2).</li>
<li><span class="mono">_get_match_pool</span> import는 Dome working-tree의 미커밋 패치에 의존(POST_TRAINING_TODO §1).</li>
</ul>

<h2>산출물</h2>
<div class="tablewrap"><table class="kv"><tbody>
<tr><td><span class="mono">best_stg1.pth</span></td><td><strong>배포/평가용</strong> — ep29 EMA(AP 0.6154), 716 MB. <span class="mono">init_report.json</span>과 함께 보관</td></tr>
<tr><td><span class="mono">last.pth</span>, <span class="mono">checkpoint0000–0029.pth</span></td><td>매 epoch 체크포인트(optimizer/EMA/scheduler 포함), 각 716 MB</td></tr>
<tr><td><span class="mono">log.txt</span> / <span class="mono">train_run.log</span> / <span class="mono">launcher_resume.log</span> / <span class="mono">launcher_launch2.log</span></td><td>epoch JSON(ep1–29) / launch #2+#3 stdout 전체 / launch #3만 / launch #2(ENOSPC 기록)</td></tr>
<tr><td><span class="mono">det/reports/260827-p4-strict-local-30ep/</span></td><td><span class="mono">REPORT.md</span>, <span class="mono">epoch_metrics.csv</span>, <span class="mono">epoch_table.md</span>, <span class="mono">iter_loss.csv</span>, <span class="mono">summary.json</span>, <span class="mono">fig1–5*.png</span>, <span class="mono">make_report.py</span>(재생성). 사본: <span class="mono">&lt;run&gt;/report/</span></td></tr>
</tbody></table></div>
<p>런 디렉터리: <span class="mono">/home/work/.mnt/DET_RESULT/lsp_detr/[combined]LSP-T_strict-local_H100x8_hf5class-ft_30ep/</span></p>

<h2>다음 단계</h2>
<p>POST_TRAINING_TODO.md 순서대로 — §1 Dome 의존성 고정 → §2 resume 아암 가드(movable-reference 런 전 필수) → <span class="mono">det/inference.py TILING=off</span>로 ep29 단일 프로세스 재평가(정확한 area-subset) → P5 Dome 기준선(같은 bundle, areafix json, dummy 제거, top-2000) 또는 movable-reference 아암 런. test_eval 킷도 GPU 제약 없이 실행 가능.</p>
</div>
"""
open(OUT, "w").write(page)
print(OUT, len(page) // 1024, "KB")
