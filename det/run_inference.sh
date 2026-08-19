#!/bin/bash
# ============================================================================
# LSP-DETR (bbox port) inference launcher  -  counterpart of AIVIS-Dome-DETR/run_*.sh
#
# Usage:
#   ./run_inference.sh /path/to/image_or_dir_or_coco.json          (positional = INPUT)
#   INPUT=/path/to/dir OUTPUT_DIR=/path/out ./run_inference.sh
#   RESUME=.../checkpoint0003.pth GPU=4 BATCH_SIZE=4 VISUALIZE=True ./run_inference.sh /path/to/dir
#
# Presets:
#   # 1) val split with COCO evaluation (reproduces the trainer's log.txt numbers for RESUME)
#   TILING=off INPUT=/home/work/.mnt/combined_all_v1_bundle/val_coco.json \
#   IMG_FOLDER=/home/work/.mnt/combined_all_v1_bundle \
#   GT_JSON=/home/work/.mnt/combined_all_v1_bundle/val_coco.json ./run_inference.sh
#   # 2) quick smoke on 20 val images with overlays
#   LIMIT=20 VISUALIZE=True ./run_inference.sh /home/work/.mnt/combined_all_v1_bundle/val/breast_HER2
#   # 3) TILING=auto (default) = full coverage: centred 1536-tile grid, stride 672, centre-672 ownership, reflect padding
#   #    (the model only fires in the centre 672 of a 1536 input). TILING=off = one forward per image (eval parity only).
#
# Every knob below can be overridden from the environment (KEY=value ./run_inference.sh ...).
# ============================================================================

# ---------------------------------------------------------------------------- environment
unset PYTORCH_CUDA_ALLOC_CONF
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

DET_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
PY=${PY:-/home/work/miniconda3/envs/dome/bin/python}   # default `python` on this box has no torch
PYTHON_SCRIPT="$DET_DIR/inference.py"

# ---------------------------------------------------------------------------- configuration
GPU=${GPU:-0}                                   # -> CUDA_VISIBLE_DEVICES (avoid GPU 5 while the 8-GPU training runs)
DEVICE=${DEVICE:-cuda}

CONFIG=${CONFIG:-"$DET_DIR/configs/LSP-T-combined.yml"}
RUN_DIR=${RUN_DIR:-"$DET_DIR/output/[combined]LSP-T_strict-local_H100x8_hf5class-ft_30ep"}
RESUME=${RESUME:-"$RUN_DIR/best_stg1.pth"}     # best (val AP) checkpoint of the running P4 job
WEIGHTS=${WEIGHTS:-ema}                          # ema | model  (trainer evaluates the EMA weights)
CENTER_MODE=${CENTER_MODE:-""}                   # empty -> taken from init_report.json next to the checkpoint (strict-local | movable-reference)

INPUT=${INPUT:-$1}                               # image | directory | *.txt | COCO json
[ $# -gt 0 ] && shift                            # remaining positionals are passed verbatim to inference.py
IMG_FOLDER=${IMG_FOLDER:-""}                     # root for relative paths in a COCO json / txt list
GT_JSON=${GT_JSON:-""}                           # COCO GT json -> AitodCocoEvaluator (same metric as training)
LIMIT=${LIMIT:-""}                               # first N images only

OUTPUT_DIR=${OUTPUT_DIR:-"$DET_DIR/output/inference/$(basename "$(dirname "$RESUME")")_$(basename "${RESUME%.pth}")_$(date +%y%m%d_%H%M%S)"}
OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-""}

BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-4}
CONF_THRESHOLD=${CONF_THRESHOLD:-0.5}
NMS_IOU_THRESHOLD=${NMS_IOU_THRESHOLD:-""}       # empty -> yml value (0.7, class-agnostic)
NUM_TOP_QUERIES=${NUM_TOP_QUERIES:-""}           # empty -> yml value (2000)
INCLUDE_NON_TUMOR=${INCLUDE_NON_TUMOR:-True}     # False -> Tumor (class 1) only

TILING=${TILING:-auto}                           # auto | on (centred tile grid, full coverage) | off (one forward/image: eval parity, centre-only)
PATCH_SIZE=${PATCH_SIZE:-1536}                   # LSP-DETR input size (tiles)
STEP_SIZE=${STEP_SIZE:-672}
FILTER_SIZE=${FILTER_SIZE:-672}
TILE_MERGE_NMS=${TILE_MERGE_NMS:-""}             # e.g. 0.5 -> class-agnostic NMS across merged tiles

VISUALIZE=${VISUALIZE:-False}
VIS_LIMIT=${VIS_LIMIT:-""}
AMP=${AMP:-none}                                 # none | bf16 | fp16   (none == training/eval numerics)
TF32=${TF32:-keep}                               # keep | off | on
EXTRA_ARGS=${EXTRA_ARGS:-""}                     # extra flags (whitespace-split, no quoting) - or append them after INPUT on the command line

# ---------------------------------------------------------------------------- validation
if [ -z "$INPUT" ]; then
    echo "Error: input required.";  echo "Usage: $0 /path/to/image_or_dir_or_coco.json   (or INPUT=... $0)"; exit 1
fi
# absolutise every path knob (validation and python must see the same paths regardless of cwd)
INPUT=$(readlink -m -- "$INPUT"); OUTPUT_DIR=$(readlink -m -- "$OUTPUT_DIR"); RESUME=$(readlink -m -- "$RESUME"); CONFIG=$(readlink -m -- "$CONFIG")
[ -n "$GT_JSON" ] && GT_JSON=$(readlink -m -- "$GT_JSON")
[ -n "$IMG_FOLDER" ] && IMG_FOLDER=$(readlink -m -- "$IMG_FOLDER")
if [ ! -e "$INPUT" ]; then echo "Error: input not found: $INPUT"; exit 1; fi
if [ ! -f "$RESUME" ]; then echo "Error: checkpoint not found: $RESUME"; exit 1; fi
if [ ! -f "$CONFIG" ]; then echo "Error: config not found: $CONFIG"; exit 1; fi
if [ -n "$GT_JSON" ] && [ ! -f "$GT_JSON" ]; then echo "Error: GT json not found: $GT_JSON"; exit 1; fi
if [ -n "$IMG_FOLDER" ] && [ ! -d "$IMG_FOLDER" ]; then echo "Error: img folder not found: $IMG_FOLDER"; exit 1; fi
if [ ! -f "$PYTHON_SCRIPT" ]; then echo "Error: $PYTHON_SCRIPT not found"; exit 1; fi
if [ ! -x "$PY" ]; then echo "Error: python not found: $PY (set PY=...)"; exit 1; fi
if [ "$GPU" = "5" ] && nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 5 2>/dev/null | awk '{exit !($1>60000)}'; then
    echo "Warning: GPU 5 has >60 GB in use (training rank); inference may OOM there."
fi

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/inference${OUTPUT_SUFFIX}.log"

echo "=========================================="
echo "LSP-DETR inference"
echo "=========================================="
echo "Config:      $CONFIG"
echo "Checkpoint:  $RESUME  (weights=$WEIGHTS, center_mode=${CENTER_MODE:-auto from init_report.json})"
echo "Input:       $INPUT"
[ -n "$IMG_FOLDER" ] && echo "Img folder:  $IMG_FOLDER"
[ -n "$GT_JSON" ]    && echo "GT json:     $GT_JSON  (COCO eval on)"
echo "Output:      $OUTPUT_DIR"
echo "GPU:         $GPU  batch=$BATCH_SIZE  amp=$AMP  tf32=$TF32"
echo "Thresholds:  conf=$CONF_THRESHOLD nms_iou=${NMS_IOU_THRESHOLD:-yml} topk=${NUM_TOP_QUERIES:-yml} include_non_tumor=$INCLUDE_NON_TUMOR"
echo "Tiling:      $TILING  patch=$PATCH_SIZE step=$STEP_SIZE filter=$FILTER_SIZE merge_nms=${TILE_MERGE_NMS:-none}"
echo "Log:         $LOG_FILE"
echo "=========================================="

# ---------------------------------------------------------------------------- run
ARGS=(
    --config "$CONFIG"
    --resume "$RESUME"
    --weights "$WEIGHTS"
    --input "$INPUT"
    --output-dir "$OUTPUT_DIR"
    --output-suffix "$OUTPUT_SUFFIX"
    --gpu "$GPU"
    --device "$DEVICE"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --conf-threshold "$CONF_THRESHOLD"
    --include-non-tumor "$INCLUDE_NON_TUMOR"
    --tiling "$TILING"
    --patch-size "$PATCH_SIZE"
    --step-size "$STEP_SIZE"
    --filter-size "$FILTER_SIZE"
    --visualize "$VISUALIZE"
    --amp "$AMP"
    --tf32 "$TF32"
)
[ -n "$IMG_FOLDER" ]        && ARGS+=(--img-folder "$IMG_FOLDER")
[ -n "$GT_JSON" ]           && ARGS+=(--gt-json "$GT_JSON")
[ -n "$LIMIT" ]             && ARGS+=(--limit "$LIMIT")
[ -n "$NMS_IOU_THRESHOLD" ] && ARGS+=(--nms-iou-threshold "$NMS_IOU_THRESHOLD")
[ -n "$NUM_TOP_QUERIES" ]   && ARGS+=(--num-top-queries "$NUM_TOP_QUERIES")
[ -n "$TILE_MERGE_NMS" ]    && ARGS+=(--tile-merge-nms "$TILE_MERGE_NMS")
[ -n "$VIS_LIMIT" ]         && ARGS+=(--vis-limit "$VIS_LIMIT")
[ -n "$CENTER_MODE" ]       && ARGS+=(-u "LSPDetrDetection.center_mode=$CENTER_MODE")

cd "$DET_DIR"   # all paths are absolute by now; cwd only matters for relative EXTRA_ARGS values
"$PY" -u "$PYTHON_SCRIPT" "${ARGS[@]}" $EXTRA_ARGS "$@" 2>&1 | tee "$LOG_FILE"
exit_code=${PIPESTATUS[0]}

if [ $exit_code -ne 0 ]; then
    echo "Error: inference failed (exit code: $exit_code) - see $LOG_FILE"
    exit $exit_code
fi
echo "Completed. Results in $OUTPUT_DIR"
