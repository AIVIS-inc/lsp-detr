#!/usr/bin/env python
"""Build the training record for the P4 strict-local 30-epoch LSP-DETR run:
CSV tables, PNG figures, REPORT.md. Pure log parsing — no model code."""
import csv, json, os, re, sys, datetime as dt
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

RUN = "/home/work/.mnt/DET_RESULT/lsp_detr/[combined]LSP-T_strict-local_H100x8_hf5class-ft_30ep"
OUT = "/home/work/tksong/lsp-detr/det/reports/260827-p4-strict-local-30ep"
os.makedirs(OUT, exist_ok=True)
ITERS_PER_EPOCH = 6601

# ---------- palette (dataviz reference, validated) ----------
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURF, TXT, TXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": TXT, "axes.labelcolor": TXT2, "xtick.color": TXT2, "ytick.color": TXT2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2, "font.family": "DejaVu Sans",
})

# ---------- 1. log.txt (per-epoch, ep1..29) ----------
rows = []
for l in open(os.path.join(RUN, "log.txt")):
    l = l.strip()
    if not l:
        continue
    rows.append(json.loads(l))
by_ep = {r["epoch"]: r for r in rows}
assert sorted(by_ep) == list(range(1, 30)), sorted(by_ep)

# ---------- 2. train_run.log: iteration lines, epoch/eval times, COCO blocks (incl. ep0) ----------
txt = open(os.path.join(RUN, "train_run.log"), errors="replace").read().splitlines()
it_re = re.compile(r"^Epoch: \[(\d+)\]  \[\s*(\d+)/(\d+)\]  eta: \S+  lr: (\S+)  loss: (\S+) \((\S+)\)")
iters = []
for l in txt:
    m = it_re.match(l)
    if m:
        e, i, n, lr, loss_med, loss_avg = m.groups()
        iters.append((int(e), int(i), int(n), float(lr), float(loss_med), float(loss_avg)))
from collections import Counter
cnt = Counter(e for e, *_ in iters)
assert sorted(cnt) == list(range(30)), sorted(cnt)
print("iteration lines:", len(iters), "missing per epoch:", {e: 67 - c for e, c in cnt.items() if c != 67})
ep_time = {}
for l in txt:
    m = re.match(r"^Epoch: \[(\d+)\] Total time: (\d+):(\d+):(\d+) \(([\d.]+) s / it\)", l)
    if m:
        e, h, mi, s, spi = m.groups()
        ep_time[int(e)] = (int(h) * 3600 + int(mi) * 60 + int(s), float(spi))
assert sorted(ep_time) == list(range(30))
eval_time = []
for l in txt:
    m = re.match(r"^Test: Total time: (\d+):(\d+):(\d+) \(([\d.]+) s / it\)", l)
    if m:
        h, mi, s, spi = m.groups()
        eval_time.append(int(h) * 3600 + int(mi) * 60 + int(s))
assert len(eval_time) == 30, len(eval_time)
# COCO blocks: 13 lines each, in epoch order 0..29
coco_blocks, cur = [], []
for l in txt:
    m = re.match(r"^ Average (Precision|Recall)\s+\(A[PR]\) @\[.*\] = ([\d.]+)", l)
    if m:
        cur.append(float(m.group(2)))
        if len(cur) == 13:
            coco_blocks.append(cur); cur = []
assert len(coco_blocks) == 30, len(coco_blocks)
# cross-check ep1..29 against log.txt (3-decimal print vs full precision)
for e in range(1, 30):
    a = np.array(coco_blocks[e]); b = np.array(by_ep[e]["test_coco_eval_bbox"])
    assert np.all(np.abs(a - b) <= 0.0006), (e, a, b)
coco0 = coco_blocks[0]
COCO_NAMES = ["AP", "AP50", "AP_verytiny", "AP_tiny", "AP_small", "AP_medium",
              "AR@500", "AR@1000", "AR@2000", "AR_verytiny", "AR_tiny", "AR_small", "AR_medium"]
# NOTE: log.txt has 13 entries whose index 2 is AP75 in the standard COCO layout? Check ordering below.
# Dome's AITOD evaluator prints: AP, AP50, AP_verytiny, AP_tiny, AP_small, AP_medium, AR500, AR1000, AR2000, AR_vt, AR_t, AR_s, AR_m
# while log.txt test_coco_eval_bbox = evaluator.stats (same order). Earlier turns labelled index 2 "AP75" — verify:
print("ep29 printed block:", coco_blocks[29])
print("ep29 log.txt      :", by_ep[29]["test_coco_eval_bbox"])

# ---------- 3. checkpoint mtimes ----------
ck_mtime = {}
for f in os.listdir(RUN):
    m = re.match(r"checkpoint(\d{4})\.pth$", f)
    if m:
        ck_mtime[int(m.group(1))] = dt.datetime.fromtimestamp(os.path.getmtime(os.path.join(RUN, f)))
best_mtime = dt.datetime.fromtimestamp(os.path.getmtime(os.path.join(RUN, "best_stg1.pth")))

# ---------- 4. per-epoch table ----------
E = list(range(30))
def stat(e, k):
    return coco_blocks[e][k] if e == 0 else by_ep[e]["test_coco_eval_bbox"][k]
AP = np.array([stat(e, 0) for e in E]); AP50 = np.array([stat(e, 1) for e in E])
IDX2 = np.array([stat(e, 2) for e in E]); AR2000 = np.array([stat(e, 8) for e in E])
LOSS = np.array([np.nan] + [by_ep[e]["train_loss"] for e in range(1, 30)])
LR = np.array([np.nan] + [by_ep[e]["train_lr"] for e in range(1, 30)])
# ep0 train loss / lr from iteration log (final running average of epoch 0)
ep0_last = [x for x in iters if x[0] == 0][-1]
LOSS[0] = ep0_last[5]; LR[0] = by_ep[1]["train_lr"]  # iteration log prints lr with 6 decimals (0.000013); true value 1.25e-5
comp = {}
for k in ("loss_vfl", "loss_giou", "loss_bbox", "loss_vfl_aux_0", "loss_giou_aux_0", "loss_bbox_aux_0"):
    comp[k] = np.array([np.nan] + [by_ep[e]["train_" + k] for e in range(1, 30)])

table_path = os.path.join(OUT, "epoch_metrics.csv")
with open(table_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["epoch", "lr", "train_loss", "train_loss_vfl", "train_loss_giou", "train_loss_bbox",
                "AP", "AP50", "AP_verytiny_8gpu_polluted", "AR@2000"] + [f"coco_{i}" for i in range(13)] +
               ["train_time_s", "train_s_per_it", "eval_time_s", "checkpoint_mtime"])
    for e in E:
        w.writerow([e, f"{LR[e]:.3e}", f"{LOSS[e]:.4f}",
                    "" if e == 0 else f"{comp['loss_vfl'][e]:.4f}", "" if e == 0 else f"{comp['loss_giou'][e]:.4f}",
                    "" if e == 0 else f"{comp['loss_bbox'][e]:.4f}",
                    f"{AP[e]:.4f}", f"{AP50[e]:.4f}", f"{IDX2[e]:.4f}", f"{AR2000[e]:.4f}"] +
                   [f"{stat(e, i):.4f}" for i in range(13)] +
                   [ep_time[e][0], ep_time[e][1], eval_time[e], ck_mtime[e].strftime("%Y-%m-%d %H:%M")])

with open(os.path.join(OUT, "iter_loss.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["epoch", "iter", "global_epoch", "lr", "loss_window_median", "loss_epoch_running_avg"])
    for e, i, n, lr, lm, la in iters:
        w.writerow([e, i, f"{e + i / n:.4f}", lr, lm, la])

# ---------- 5. figures ----------
def lr_drop(ax, y=None):
    ax.axvline(24, color=TXT2, lw=1, ls=(0, (4, 3)))
    ax.text(24.15, ax.get_ylim()[1] if y is None else y, "LR ×0.8 (ep24)", color=TXT2, fontsize=8.5, va="top")

# fig1: train loss, iteration-level + epoch mean
fig, ax = plt.subplots(figsize=(9, 4.2), dpi=160)
x = np.array([e + i / n for e, i, n, *_ in iters]); y = np.array([lm for *_, lm, _ in iters])
ax.plot(x, y, color=S1, lw=0.8, alpha=0.55, label="iteration (20-step window median, every 100 it)")
xe = np.arange(30) + 1.0
ax.plot(xe, LOSS, color=S1, lw=2, marker="o", ms=5, mec=SURF, mew=1, label="epoch mean (log.txt)")
ax.set_xlim(0, 30.3); ax.xaxis.set_major_locator(MultipleLocator(2)); ax.set_xlabel("epoch (position within epoch = fraction)")
ax.set_ylabel("total train loss (final + 5 aux layers)")
ax.set_title("Train loss — falls monotonically all 30 epochs (11.6 → 9.04)")
for e in (0, 9, 19, 29):
    ax.annotate(f"{LOSS[e]:.2f}", (e + 1, LOSS[e]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.5, color=TXT2)
lr_drop(ax, 15.2)
ax.legend(loc="upper center", bbox_to_anchor=(0.45, 0.98), fontsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_train_loss.png")); plt.close(fig)

# fig2: loss components (final layer vs first aux layer), small multiples
fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), dpi=160)
for ax, (k, title) in zip(axs, (("loss_vfl", "VFL (classification)"), ("loss_giou", "GIoU"), ("loss_bbox", "L1 box"))):
    ax.plot(xe, comp[k], color=S1, marker="o", ms=4, mec=SURF, mew=0.8, label="final decoder layer")
    ax.plot(xe, comp[k + "_aux_0"], color=S2, lw=1.5, ls=(0, (3, 2)), label="aux layer 0")
    ax.set_title(title); ax.set_xlim(0.5, 30); ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.text(29.6, comp[k][29], f" {comp[k][29]:.3f}", color=TXT, fontsize=8, va="center")
    ax.text(29.6, comp[k + "_aux_0"][29], f" {comp[k + '_aux_0'][29]:.3f}", color=TXT2, fontsize=8, va="center")
    ax.axvline(24, color=TXT2, lw=1, ls=(0, (4, 3)))
axs[0].set_ylabel("epoch-mean loss"); axs[1].set_xlabel("epoch")
axs[0].legend(loc="upper right", fontsize=8)
fig.suptitle("Loss components (epoch means, ep1–29) — every term still decreasing; aux-0 tracks final layer", fontweight="bold", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_loss_components.png")); plt.close(fig)

# fig3: validation metrics
fig, ax = plt.subplots(figsize=(9, 4.6), dpi=160)
series = [("AP50", AP50, S1), ("AR@2000", AR2000, S3), ("AP@[.5:.95]", AP, S2)]
for name, arr, c in series:
    ax.plot(E, arr, color=c, marker="o", ms=4.5, mec=SURF, mew=0.8, label=name)
    ax.text(29.4, arr[29], f" {name} {arr[29]:.4f}", color=TXT, fontsize=8.5, va="center")
ax.set_xlim(-0.5, 34.5); ax.set_ylim(0.38, 0.92); ax.xaxis.set_major_locator(MultipleLocator(2))
ax.set_xlabel("epoch (EMA weights, val 9,853 images, top-2000 + NMS 0.7)"); ax.set_ylabel("COCO metric")
ax.set_title("Validation (EMA) — AP rises every epoch 0.429 → 0.615; AP50 flat from ep14, moves only after the LR drop")
lr_drop(ax, 0.91)
ax.legend(loc="lower right", fontsize=8.5, ncol=3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_val_metrics.png")); plt.close(fig)

# fig4: per-epoch deltas
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), dpi=160)
d_ap = np.diff(AP) * 100; d_ap50 = np.diff(AP50) * 100
for ax, d, name, c in ((axs[0], d_ap, "ΔAP", S2), (axs[1], d_ap50, "ΔAP50", S1)):
    ax.bar(np.arange(1, 30), d, width=0.72, color=c, edgecolor=SURF, linewidth=1)
    ax.set_title(f"{name} per epoch (pp)"); ax.set_xlim(0.3, 29.7); ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.axhline(0, color=TXT2, lw=0.8); ax.axvline(24, color=TXT2, lw=1, ls=(0, (4, 3)))
    for e in (1, 5, 10, 15, 20, 23, 25, 29):
        ax.text(e, d[e - 1], f"{d[e - 1]:+.2f}", ha="center", va="bottom", fontsize=7.5, color=TXT2)
axs[0].set_ylabel("percentage points"); axs[0].set_xlabel("epoch"); axs[1].set_xlabel("epoch")
fig.suptitle("Per-epoch gains — AP down to +0.03–0.10 pp/ep; AP50 ≈ 0 since ep14 except the ep25 LR-drop bump (+0.34 pp)",
             fontweight="bold", fontsize=10.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_epoch_deltas.png")); plt.close(fig)

# fig5: epoch time
fig, ax = plt.subplots(figsize=(9, 3.6), dpi=160)
spi = np.array([ep_time[e][1] for e in E]); th = np.array([ep_time[e][0] / 3600 for e in E])
ax.bar(E, spi, width=0.72, color=S1, edgecolor=SURF, linewidth=1)
for e in (0, 1, 12, 24, 29):
    ax.text(e, spi[e] + 0.05, f"{spi[e]:.2f}", ha="center", fontsize=8, color=TXT2)
ax.set_xlim(-0.6, 29.6); ax.xaxis.set_major_locator(MultipleLocator(2)); ax.set_ylim(0, 4.6)
ax.set_xlabel("epoch"); ax.set_ylabel("train s / iteration (8 × H100, batch 8)")
ax.set_title(f"Step time drifted 2.83 → 4.07 s/it — train {th.min():.1f}–{th.max():.1f} h/epoch + eval ≈ {np.mean(eval_time)/60:.0f} min (CPU matcher-bound)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_epoch_time.png")); plt.close(fig)

# ---------- 6. summary numbers for the report ----------
summary = dict(
    ep0=dict(AP=AP[0], AP50=AP50[0], idx2=IDX2[0], AR=AR2000[0]),
    ep29=dict(AP=AP[29], AP50=AP50[29], idx2=IDX2[29], AR=AR2000[29]),
    total_train_s=sum(v[0] for v in ep_time.values()), total_eval_s=sum(eval_time),
    mean_ep_h=np.mean([v[0] for v in ep_time.values()]) / 3600, mean_eval_min=np.mean(eval_time) / 60,
    spi_min=spi.min(), spi_max=spi.max(),
    best_mtime=best_mtime.strftime("%Y-%m-%d %H:%M"), ck0=ck_mtime[0].strftime("%Y-%m-%d %H:%M"), ck29=ck_mtime[29].strftime("%Y-%m-%d %H:%M"),
    dAP_last3=[float(x) for x in np.diff(AP)[-3:]], dAP50_ep25=float(AP50[25] - AP50[24]),
    ap50_plateau=(float(AP50[14:25].min()), float(AP50[14:25].max())),
    coco29=coco_blocks[29], coco0=coco0,
)
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2, default=float)
# markdown epoch table
md = ["| ep | lr | train loss | VFL | GIoU | L1 | AP | AP50 | AP_verytiny* | AR@2000 | ΔAP (pp) | s/it | train h | eval min | ckpt time |",
      "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
for e in E:
    dap = "" if e == 0 else f"{(AP[e]-AP[e-1])*100:+.2f}"
    vf = "" if e == 0 else f"{comp['loss_vfl'][e]:.3f}"; gi = "" if e == 0 else f"{comp['loss_giou'][e]:.3f}"; l1 = "" if e == 0 else f"{comp['loss_bbox'][e]:.4f}"
    md.append(f"| {e} | {LR[e]:.2e} | {LOSS[e]:.3f} | {vf} | {gi} | {l1} | {AP[e]:.4f} | {AP50[e]:.4f} | {IDX2[e]:.4f} | {AR2000[e]:.4f} | {dap} | {ep_time[e][1]:.2f} | {ep_time[e][0]/3600:.2f} | {eval_time[e]/60:.0f} | {ck_mtime[e]:%m-%d %H:%M} |")
open(os.path.join(OUT, "epoch_table.md"), "w").write("\n".join(md) + "\n")
print(json.dumps(summary, indent=1, default=float))
print("written to", OUT)
