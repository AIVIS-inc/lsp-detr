#!/bin/bash
# LSP-DETR-T (hf-5class fine-tune) distributed training - clone of AIVIS-Dome-DETR/dist_train.sh.
# Only CONFIG / OUTPUT_DIR / entry point (det/train.py) differ; env (OMP_NUM_THREADS=1, DOME_MATCH_THREADS=6,
# PYTORCH_CUDA_ALLOC_CONF unset) and the CLI overrides are identical to the Dome HER2 run.
#
# Usage:  bash det/scripts/dist_train_lsp.sh                       # 8 GPU, 30 ep, seed 0, 2-class TNT bundle
#         CONFIG=configs/LSP-T-her2.yml bash det/scripts/dist_train_lsp.sh        # 5-class HER2 (new_merged_all)
#         NUM_GPUS=2 GPU_IDS=0,1 END_EPOCHS=1 bash det/scripts/dist_train_lsp.sh   # smoke
#         EXTRA_UPDATES="LSPDetrDetection.center_mode=movable-reference" bash det/scripts/dist_train_lsp.sh
#         RESUME=<run>/last.pth bash det/scripts/dist_train_lsp.sh              # continue a stopped run
#   On another machine (bare clone, see det/README.md "Setup on a new machine"): the dataset paths baked into
#   configs/dataset/*.yml and the output root are env-overridable, and DRY_RUN=1 prints the resolved command:
#         TRAIN_IMG_DIR=/data/bundle TRAIN_ANN=/data/bundle/train_coco_areafix.json \
#         VAL_IMG_DIR=/data/bundle   VAL_ANN=/data/bundle/val_coco.json \
#         OUTPUT_ROOT=/big/disk/lsp_detr DRY_RUN=1 bash det/scripts/dist_train_lsp.sh
#   Env knobs: NUM_GPUS GPU_IDS END_EPOCHS SEED CONFIG ARM OUTPUT_ROOT OUTPUT_DIR PYTHON TF32=keep|off|on RESUME
#              EXTRA_UPDATES="key=val ..." TRAIN_IMG_DIR TRAIN_ANN VAL_IMG_DIR VAL_ANN DRY_RUN MASTER_PORT DOME_MATCH_THREADS

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DET_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${DET_DIR}"

# Configuration (env-overridable)
MASTER_PORT=${MASTER_PORT:-7789}
NUM_GPUS=${NUM_GPUS:-8}
GPU_IDS=${GPU_IDS:-"0,1,2,3,4,5,6,7"}
TRAIN_BATCH_SIZE_PER_GPU=${TRAIN_BATCH_SIZE_PER_GPU:-1}
TOTAL_TRAIN_BATCH_SIZE=$((NUM_GPUS * TRAIN_BATCH_SIZE_PER_GPU))
TOTAL_VAL_BATCH_SIZE=${TOTAL_VAL_BATCH_SIZE:-${NUM_GPUS}}
CONFIG=${CONFIG:-"configs/LSP-T-combined.yml"}
END_EPOCHS=${END_EPOCHS:-30}
ARM=${ARM:-strict-local}
# Output root. 30 epochs at checkpoint_freq=1 need >23 GB (716 MB per checkpoint): on the training box the NFS
# mount is used (/home/work is a 49 GB loop device; launch #2 died there with ENOSPC on 2026-08-18) and det/output
# is a symlink to it; anywhere else the default is <det>/output. Override with OUTPUT_ROOT= or OUTPUT_DIR=.
if [ -z "${OUTPUT_ROOT:-}" ]; then
    if [ -d /home/work/.mnt/DET_RESULT/lsp_detr ]; then
        OUTPUT_ROOT=/home/work/.mnt/DET_RESULT/lsp_detr
    else
        OUTPUT_ROOT="${DET_DIR}/output"
    fi
fi
# Tag derived from CONFIG (review 2026-09-04): the old literal "[combined]" default made a CONFIG-only HER2
# launch write into the COMPLETED TNT run's directory and overwrite its checkpoints.
CFG_TAG=$(basename "${CONFIG%.yml}"); CFG_TAG=${CFG_TAG#LSP-T-}
OUTPUT_DIR=${OUTPUT_DIR:-"${OUTPUT_ROOT}/[${CFG_TAG}]LSP-T_${ARM}_H100x${NUM_GPUS}_hf5class-ft_${END_EPOCHS}ep"}
SEED=${SEED:-0}
# Python: $PYTHON, else the first interpreter that can import torch - the training box's conda env, then PATH
# (same rule as run_inference.sh).
PYTHON=${PYTHON:-}
if [ -z "${PYTHON}" ]; then
    for cand in /home/work/miniconda3/envs/dome/bin/python "$(command -v python3 || true)" "$(command -v python || true)"; do
        [ -n "${cand}" ] && [ -x "${cand}" ] || continue
        if (unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF; "${cand}" -c "import torch" >/dev/null 2>&1); then
            PYTHON="${cand}"; break
        fi
    done
fi
EXTRA_UPDATES=${EXTRA_UPDATES:-""}
TF32=${TF32:-keep}   # keep = machine default (TF32 on via TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1, as the Dome runs); off = pure FP32
RESUME=${RESUME:-""}  # path to last.pth to resume a stopped run (solver restarts at last_epoch+1)
DRY_RUN=${DRY_RUN:-0} # 1 = print the resolved command and exit (no GPU check, nothing created)
# Dataset paths (optional): override the absolute paths in configs/dataset/*.yml without editing them.
# Plain paths only (they are passed as `-u key=value` and parsed as YAML).
TRAIN_IMG_DIR=${TRAIN_IMG_DIR:-""}
TRAIN_ANN=${TRAIN_ANN:-""}
VAL_IMG_DIR=${VAL_IMG_DIR:-""}
VAL_ANN=${VAL_ANN:-""}

# Preflight checks
if [ "${DRY_RUN}" != "1" ]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "Error: nvidia-smi was not found." >&2
        exit 1
    fi
    AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    if (( AVAILABLE_GPUS < NUM_GPUS )); then
        echo "Error: ${NUM_GPUS} GPUs are required, but only ${AVAILABLE_GPUS} were detected." >&2
        exit 1
    fi
fi
if [ -z "${PYTHON}" ] || [ ! -x "${PYTHON}" ]; then
    echo "Error: no python with torch found (set PYTHON=/path/to/python; see det/README.md for the environment)" >&2
    exit 1
fi
if [ ! -f "${CONFIG}" ]; then
    echo "Error: config not found: ${CONFIG} (relative to ${DET_DIR})" >&2
    exit 1
fi
HF5CLASS_CKPT="${DET_DIR}/../hf-5class/model.safetensors"
if [ ! -f "${HF5CLASS_CKPT}" ] && [[ "${EXTRA_UPDATES}" != *pretrained=* ]]; then
    echo "Error: initial checkpoint missing: ${HF5CLASS_CKPT}. Run: ${PYTHON} ${DET_DIR}/scripts/fetch_hf5class.py" >&2
    exit 1
fi

echo "=========================================="
echo "LSP-DETR Distributed Training (hf-5class fine-tune)"
echo "=========================================="
echo "Arm: ${ARM}"
echo "GPUs: ${NUM_GPUS} (${GPU_IDS})"
echo "Train batch: ${TOTAL_TRAIN_BATCH_SIZE} total (${TRAIN_BATCH_SIZE_PER_GPU} per GPU)"
echo "Validation batch: ${TOTAL_VAL_BATCH_SIZE} total"
echo "Config: ${CONFIG}"
echo "Output: ${OUTPUT_DIR}"
echo "Python: ${PYTHON}"
echo "End epochs: ${END_EPOCHS}"
echo "Seed: ${SEED}"
echo "Extra updates: ${EXTRA_UPDATES}"
echo "Dataset overrides: train=${TRAIN_IMG_DIR:-<yml>} / ${TRAIN_ANN:-<yml>}  val=${VAL_IMG_DIR:-<yml>} / ${VAL_ANN:-<yml>}"
echo "TF32 policy: ${TF32}"
echo "=========================================="
echo ""

# Enable intermediate visualization results (for debugging)
export SAVE_INTERMEDIATE_VISUALIZE_RESULT=False

# Avoid PyTorch AllocatorConfig parser aborts from inherited shell settings.
unset PYTORCH_CUDA_ALLOC_CONF

# Reduce NCCL logging verbosity (ERROR=3, WARN=2, INFO=1, DEBUG=0)
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=INIT,COLL

# Solve the per-layer Hungarian matches on a thread pool inside the loss (result-identical speedup).
export DOME_MATCH_THREADS=${DOME_MATCH_THREADS:-6}

UPDATES=(epoches=${END_EPOCHS}
         train_dataloader.total_batch_size=${TOTAL_TRAIN_BATCH_SIZE}
         val_dataloader.total_batch_size=${TOTAL_VAL_BATCH_SIZE}
         LSPDetrDetection.center_mode=${ARM})
# dataset overrides first, so an explicit EXTRA_UPDATES for the same key still wins
if [ -n "${TRAIN_IMG_DIR}" ]; then UPDATES+=("train_dataloader.dataset.dataset.img_folder=${TRAIN_IMG_DIR}"); fi
if [ -n "${TRAIN_ANN}" ];     then UPDATES+=("train_dataloader.dataset.dataset.ann_file=${TRAIN_ANN}"); fi
if [ -n "${VAL_IMG_DIR}" ];   then UPDATES+=("val_dataloader.dataset.img_folder=${VAL_IMG_DIR}"); fi
if [ -n "${VAL_ANN}" ];       then UPDATES+=("val_dataloader.dataset.ann_file=${VAL_ANN}"); fi
if [ -n "${EXTRA_UPDATES}" ]; then
    # shellcheck disable=SC2206
    UPDATES+=(${EXTRA_UPDATES})
fi
RESUME_ARGS=()
if [ -n "${RESUME}" ]; then
    if [ ! -f "${RESUME}" ]; then
        echo "Error: RESUME checkpoint not found: ${RESUME}" >&2
        exit 1
    fi
    RESUME_ARGS=(-r "${RESUME}")
    echo "Resuming from: ${RESUME}"
fi

CMD=("${PYTHON}" -m torch.distributed.run
     --standalone
     --nnodes=1
     --nproc_per_node=${NUM_GPUS}
     --master_port=${MASTER_PORT}
     train.py
     -c "${CONFIG}"
     --output-dir "${OUTPUT_DIR}"
     --seed ${SEED}
     --tf32 "${TF32}"
     "${RESUME_ARGS[@]}"
     -u "${UPDATES[@]}")

if [ "${DRY_RUN}" = "1" ]; then
    echo "[dry-run] cd ${DET_DIR}"
    printf '[dry-run] CUDA_VISIBLE_DEVICES=%q OMP_NUM_THREADS=1 DOME_MATCH_THREADS=%q' "${GPU_IDS}" "${DOME_MATCH_THREADS}"
    printf ' %q' "${CMD[@]}"; printf ' 2>&1 | tee -a %q\n' "${OUTPUT_DIR}/train_run.log"
    exit 0
fi

# refuse to clobber an existing run (fresh launches only; resume passes RESUME=)
if [ -z "${RESUME}" ] && { [ -e "${OUTPUT_DIR}/last.pth" ] || [ -e "${OUTPUT_DIR}/log.txt" ] || ls "${OUTPUT_DIR}"/checkpoint*.pth >/dev/null 2>&1; }; then
    echo "Error: OUTPUT_DIR ${OUTPUT_DIR} already contains a run (last.pth/log.txt/checkpoint*). Set RESUME=... to resume or choose another OUTPUT_DIR." >&2
    exit 1
fi
mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
OMP_NUM_THREADS=1 \
"${CMD[@]}" \
    2>&1 | tee -a "${OUTPUT_DIR}/train_run.log"

echo ""
echo "Training completed!"
