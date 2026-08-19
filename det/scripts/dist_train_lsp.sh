#!/bin/bash
# LSP-DETR-T (hf-5class fine-tune) distributed training - clone of AIVIS-Dome-DETR/dist_train.sh.
# Only CONFIG / OUTPUT_DIR / entry point (det/train.py) differ; env (OMP_NUM_THREADS=1, DOME_MATCH_THREADS=6,
# PYTORCH_CUDA_ALLOC_CONF unset) and the CLI overrides are identical to the Dome HER2 run.
#
# Usage:  bash det/scripts/dist_train_lsp.sh                       # 8 GPU, 30 ep, seed 0
#         NUM_GPUS=2 GPU_IDS=0,1 END_EPOCHS=1 bash det/scripts/dist_train_lsp.sh   # smoke
#         EXTRA_UPDATES="LSPDetrDetection.center_mode=movable-reference" bash det/scripts/dist_train_lsp.sh

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
# Outputs MUST live on the NFS mount: /home/work is a 49 GB loop device and 30 epochs of checkpoints need >23 GB
# (launch #2 died with ENOSPC on 2026-08-18). det/output is a symlink to this directory.
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/work/.mnt/DET_RESULT/lsp_detr}
OUTPUT_DIR=${OUTPUT_DIR:-"${OUTPUT_ROOT}/[combined]LSP-T_${ARM}_H100x${NUM_GPUS}_hf5class-ft_${END_EPOCHS}ep"}
SEED=${SEED:-0}
PYTHON=${PYTHON:-/home/work/miniconda3/envs/dome/bin/python}
EXTRA_UPDATES=${EXTRA_UPDATES:-""}
TF32=${TF32:-keep}   # keep = machine default (TF32 on via TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1, as the Dome runs); off = pure FP32
RESUME=${RESUME:-""}  # path to last.pth to resume a stopped run (solver restarts at last_epoch+1)

# Preflight checks
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Error: nvidia-smi was not found." >&2
    exit 1
fi
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if (( AVAILABLE_GPUS < NUM_GPUS )); then
    echo "Error: ${NUM_GPUS} GPUs are required, but only ${AVAILABLE_GPUS} were detected." >&2
    exit 1
fi
if [ ! -x "${PYTHON}" ]; then
    echo "Error: python not found at ${PYTHON}" >&2
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
echo "End epochs: ${END_EPOCHS}"
echo "Seed: ${SEED}"
echo "Extra updates: ${EXTRA_UPDATES}"
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

mkdir -p "${OUTPUT_DIR}"

UPDATES=(epoches=${END_EPOCHS}
         train_dataloader.total_batch_size=${TOTAL_TRAIN_BATCH_SIZE}
         val_dataloader.total_batch_size=${TOTAL_VAL_BATCH_SIZE}
         LSPDetrDetection.center_mode=${ARM})
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

CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
OMP_NUM_THREADS=1 \
"${PYTHON}" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    train.py \
    -c "${CONFIG}" \
    --output-dir "${OUTPUT_DIR}" \
    --seed ${SEED} \
    --tf32 "${TF32}" \
    "${RESUME_ARGS[@]}" \
    -u "${UPDATES[@]}" \
    2>&1 | tee -a "${OUTPUT_DIR}/train_run.log"

echo ""
echo "Training completed!"
