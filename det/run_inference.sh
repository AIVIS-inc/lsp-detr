#!/bin/bash
# ============================================================================
# LSP-DETR (bbox port) inference launcher  -  counterpart of AIVIS-Dome-DETR/run_*.sh
#
# Usage:
#   ./run_inference.sh /path/to/image_or_dir_or_coco.json          (positional = INPUT)
#   INPUT=/path/to/dir OUTPUT_DIR=/path/out ./run_inference.sh
#   RESUME=.../checkpoint0003.pth GPU=4 BATCH_SIZE=4 VISUALIZE=True ./run_inference.sh /path/to/dir
#   GPUS=0,1,2,3 ./run_inference.sh /path/to/slide.tif                (WSI -> .zst, auto-detected)
#
# Two modes:
#   image  (det/inference.py)  image / directory / *.txt / COCO json  -> predictions.json + counts csv
#   wsi    (det/wsi_infer.py)  whole-slide image                      -> <slide.ext>_LSPDETR-TNT.zst (+ COCO json)
#          OpenSlide formats, plus Philips .isyntax/.i2syntax via the TILs-Inference pixel-engine wrapper.
#          A WSI is streamed tile by tile at --target-mpp (0.5, the mpp this arm is run at) with the
#          Dome-DETR tumour/non-tumour geometry (patch 1536 / step 672 / filter 672) and the tiles are
#          sharded over $GPUS. MODE=auto (default) probes the input with OpenSlide; force with MODE=wsi|image.
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
#   # 4) whole-slide image on four GPUs, .zst into <repo>/results
#   GPUS=0,1,2,3 ./run_inference.sh /workspace/AIVIS-lsp-detr/wsi/129S.tif
#
# Every knob below can be overridden from the environment (KEY=value ./run_inference.sh ...).
# ============================================================================

# ---------------------------------------------------------------------------- environment
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF   # torch >= 2.9 aborts on this box's malformed value
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

DET_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
LSP_ROOT="$(dirname "$DET_DIR")"
# First interpreter that can import torch: $PY, the training box's conda env, then whatever is on PATH.
if [ -z "${PY:-}" ]; then
    for cand in /home/work/miniconda3/envs/dome/bin/python "$(command -v python3)" "$(command -v python)"; do
        [ -x "$cand" ] || continue
        if (unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF; "$cand" -c "import torch" >/dev/null 2>&1); then
            PY="$cand"; break
        fi
    done
fi
PYTHON_SCRIPT="$DET_DIR/inference.py"
WSI_SCRIPT="$DET_DIR/wsi_infer.py"

# ---------------------------------------------------------------------------- configuration
GPU=${GPU:-0}                                   # -> CUDA_VISIBLE_DEVICES (avoid GPU 5 while the 8-GPU training runs)
DEVICE=${DEVICE:-cuda}

CONFIG=${CONFIG:-"$DET_DIR/configs/LSP-T-combined.yml"}
# Canonical location of the released TNT checkpoint (NAS). Override with RUN_DIR=<dir> for a
# training run dir (or the in-repo copy at $DET_DIR/ckpt), or RESUME=<file> for one checkpoint.
RUN_DIR=${RUN_DIR:-/mnt/nas2/LSP-DETR/TNT}
RESUME=${RESUME:-"$RUN_DIR/best_stg1.pth"}       # best (val AP) checkpoint
WEIGHTS=${WEIGHTS:-ema}                          # ema | model  (trainer evaluates the EMA weights)
CENTER_MODE=${CENTER_MODE:-""}                   # empty -> taken from init_report.json next to the checkpoint (strict-local | movable-reference)

INPUT=${INPUT:-$1}                               # image | directory | *.txt | COCO json
[ $# -gt 0 ] && shift                            # remaining positionals are passed verbatim to inference.py
IMG_FOLDER=${IMG_FOLDER:-""}                     # root for relative paths in a COCO json / txt list
GT_JSON=${GT_JSON:-""}                           # COCO GT json -> AitodCocoEvaluator (same metric as training)
LIMIT=${LIMIT:-""}                               # first N images only

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

# ---------------------------------------------------------------------------- whole-slide (MODE=wsi)
MODE=${MODE:-auto}                               # auto (probe with OpenSlide) | wsi | image
GPUS=${GPUS:-$GPU}                               # WSI: comma separated ids, one worker process per GPU
TARGET_MPP=${TARGET_MPP:-0.5}                    # mpp this arm is run at (Dome-DETR tumour/non-tumour recipe)
SOURCE_MPP=${SOURCE_MPP:-""}                     # override the slide's level-0 mpp (default: read from the file)
MPP_TOLERANCE=${MPP_TOLERANCE:-""}               # empty -> 0.05: a level this close to TARGET_MPP is read as is
TAG=${TAG:-LSPDETR-TNT}                          # suffix after the slide's full file name -> 129S.tif_LSPDETR-TNT.zst
OUTPUT_NAME=${OUTPUT_NAME:-""}                   # override the whole base name (default <slide file name>_$TAG)
SEG_DOWNSAMPLE=${SEG_DOWNSAMPLE:-32}             # tissue-mask downsample
SAT_MIN=${SAT_MIN:-8}                            # minimum HSV saturation for tissue
READER_THREADS=${READER_THREADS:-8}              # tile decode threads per worker
PREFETCH=${PREFETCH:-4}                          # batches read ahead of the GPU
MAX_TILES=${MAX_TILES:-""}                       # cap the tile count (smoke tests)
SAVE_JSON=${SAVE_JSON:-True}                     # False -> only the .zst

VISUALIZE=${VISUALIZE:-False}
VIS_LIMIT=${VIS_LIMIT:-""}
AMP=${AMP:-none}                                 # none | bf16 | fp16   (none == training/eval numerics)
TF32=${TF32:-keep}                               # keep | off | on
EXTRA_ARGS=${EXTRA_ARGS:-""}                     # extra flags (whitespace-split, no quoting) - or append them after INPUT on the command line

# ---------------------------------------------------------------------------- knob sanity
# The enum knobs are plain names that the surrounding environment may already export for
# something else (this box has TF32=1), which argparse would then reject; ignore such values.
_enum() {   # _enum <var-name> <default> <allowed...>
    local name="$1" def="$2" cur a; cur="${!name}"; shift 2
    for a in "$@"; do [ "$cur" = "$a" ] && return 0; done
    echo "Warning: $name='$cur' is not one of: $* - using '$def'"
    printf -v "$name" '%s' "$def"
}
_enum MODE    auto auto wsi image
_enum WEIGHTS ema  ema model
_enum TILING  auto auto on off
_enum AMP     none none bf16 fp16
_enum TF32    keep keep off on

# ---------------------------------------------------------------------------- validation
if [ -z "$INPUT" ]; then
    echo "Error: input required.";  echo "Usage: $0 /path/to/image_or_dir_or_coco.json   (or INPUT=... $0)"; exit 1
fi
# absolutise every path knob (validation and python must see the same paths regardless of cwd)
INPUT=$(readlink -m -- "$INPUT"); RESUME=$(readlink -m -- "$RESUME"); CONFIG=$(readlink -m -- "$CONFIG")
[ -n "$GT_JSON" ] && GT_JSON=$(readlink -m -- "$GT_JSON")
[ -n "$IMG_FOLDER" ] && IMG_FOLDER=$(readlink -m -- "$IMG_FOLDER")
if [ ! -e "$INPUT" ]; then echo "Error: input not found: $INPUT"; exit 1; fi

# MODE=auto: a single file OpenSlide opens with more than one pyramid level is a WSI (a plain .tif is not).
if [ "$MODE" = "auto" ]; then
    MODE=image
    if [ -f "$INPUT" ] && [ -n "$PY" ]; then
        case "${INPUT,,}" in
            # Philips: OpenSlide cannot open these, so there is nothing to probe - wsi_infer.py
            # reads them through the pixel engine (see its ISYNTAX_EXT / _isyntax_reader).
            *.isyntax|*.i2syntax) MODE=wsi ;;
            *.svs|*.ndpi|*.mrxs|*.bif|*.scn|*.vms|*.vmu|*.tif|*.tiff|*.dcm)
                if "$PY" -c "
import sys
try:
    import openslide
    s = openslide.OpenSlide(sys.argv[1])
except Exception:
    sys.exit(1)
sys.exit(0 if s.level_count > 1 else 1)
" "$INPUT" 2>/dev/null; then MODE=wsi; fi ;;
        esac
    fi
fi
if [ "$MODE" = "wsi" ]; then
    OUTPUT_DIR=${OUTPUT_DIR:-"$LSP_ROOT/results"}
else
    OUTPUT_DIR=${OUTPUT_DIR:-"$DET_DIR/output/inference/$(basename "$(dirname "$RESUME")")_$(basename "${RESUME%.pth}")_$(date +%y%m%d_%H%M%S)"}
fi
OUTPUT_DIR=$(readlink -m -- "$OUTPUT_DIR")

if [ ! -f "$RESUME" ]; then echo "Error: checkpoint not found: $RESUME"; exit 1; fi
if [ ! -f "$CONFIG" ]; then echo "Error: config not found: $CONFIG"; exit 1; fi
if [ -n "$GT_JSON" ] && [ ! -f "$GT_JSON" ]; then echo "Error: GT json not found: $GT_JSON"; exit 1; fi
if [ -n "$IMG_FOLDER" ] && [ ! -d "$IMG_FOLDER" ]; then echo "Error: img folder not found: $IMG_FOLDER"; exit 1; fi
if [ ! -f "$PYTHON_SCRIPT" ]; then echo "Error: $PYTHON_SCRIPT not found"; exit 1; fi
if [ "$MODE" = "wsi" ] && [ ! -f "$WSI_SCRIPT" ]; then echo "Error: $WSI_SCRIPT not found"; exit 1; fi
if [ -z "$PY" ] || [ ! -x "$PY" ]; then echo "Error: no python with torch found (set PY=/path/to/python)"; exit 1; fi
if [ "$GPU" = "5" ] && nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 5 2>/dev/null | awk '{exit !($1>60000)}'; then
    echo "Warning: GPU 5 has >60 GB in use (training rank); inference may OOM there."
fi

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/inference${OUTPUT_SUFFIX}.log"

echo "=========================================="
echo "LSP-DETR inference  (mode=$MODE)"
echo "=========================================="
echo "Python:      $PY"
echo "Config:      $CONFIG"
echo "Checkpoint:  $RESUME  (weights=$WEIGHTS, center_mode=${CENTER_MODE:-auto from init_report.json})"
echo "Input:       $INPUT"
[ -n "$IMG_FOLDER" ] && echo "Img folder:  $IMG_FOLDER"
[ -n "$GT_JSON" ]    && echo "GT json:     $GT_JSON  (COCO eval on)"
echo "Output:      $OUTPUT_DIR"
if [ "$MODE" = "wsi" ]; then
    echo "GPUs:        $GPUS  batch=$BATCH_SIZE  amp=$AMP  tf32=$TF32  reader_threads=$READER_THREADS"
    echo "Thresholds:  conf=$CONF_THRESHOLD nms_iou=${NMS_IOU_THRESHOLD:-yml} topk=${NUM_TOP_QUERIES:-yml} include_non_tumor=$INCLUDE_NON_TUMOR"
    echo "Tiling:      patch=$PATCH_SIZE step=$STEP_SIZE filter=$FILTER_SIZE  target_mpp=$TARGET_MPP source_mpp=${SOURCE_MPP:-from file}"
    echo "Tissue:      seg_downsample=$SEG_DOWNSAMPLE sat_min=$SAT_MIN  max_tiles=${MAX_TILES:-all}"
    echo "Outputs:     ${OUTPUT_NAME:-$(basename "$INPUT")_$TAG}.zst (MVT/zstd)$([ "$SAVE_JSON" = "True" ] && echo " + .json (COCO)") + .meta.json"
else
    echo "GPU:         $GPU  batch=$BATCH_SIZE  amp=$AMP  tf32=$TF32"
    echo "Thresholds:  conf=$CONF_THRESHOLD nms_iou=${NMS_IOU_THRESHOLD:-yml} topk=${NUM_TOP_QUERIES:-yml} include_non_tumor=$INCLUDE_NON_TUMOR"
    echo "Tiling:      $TILING  patch=$PATCH_SIZE step=$STEP_SIZE filter=$FILTER_SIZE merge_nms=${TILE_MERGE_NMS:-none}"
fi
echo "Log:         $LOG_FILE"
echo "=========================================="

# ---------------------------------------------------------------------------- run (WSI -> .zst)
if [ "$MODE" = "wsi" ]; then
    WSI_ARGS=(
        --wsi "$INPUT"
        --config "$CONFIG"
        --resume "$RESUME"
        --weights "$WEIGHTS"
        --output-dir "$OUTPUT_DIR"
        --gpus "$GPUS"
        --device "$DEVICE"
        --batch-size "$BATCH_SIZE"
        --reader-threads "$READER_THREADS"
        --prefetch "$PREFETCH"
        --conf-threshold "$CONF_THRESHOLD"
        --include-non-tumor "$INCLUDE_NON_TUMOR"
        --patch-size "$PATCH_SIZE"
        --step-size "$STEP_SIZE"
        --filter-size "$FILTER_SIZE"
        --target-mpp "$TARGET_MPP"
        --seg-downsample "$SEG_DOWNSAMPLE"
        --sat-min "$SAT_MIN"
        --tag "$TAG"
        --amp "$AMP"
        --tf32 "$TF32"
    )
    [ -n "$OUTPUT_NAME" ]       && WSI_ARGS+=(--output-name "$OUTPUT_NAME")
    [ -n "$SOURCE_MPP" ]        && WSI_ARGS+=(--source-mpp "$SOURCE_MPP")
    [ -n "$MPP_TOLERANCE" ]     && WSI_ARGS+=(--mpp-tolerance "$MPP_TOLERANCE")
    [ -n "$MAX_TILES" ]         && WSI_ARGS+=(--max-tiles "$MAX_TILES")
    [ -n "$NMS_IOU_THRESHOLD" ] && WSI_ARGS+=(--nms-iou-threshold "$NMS_IOU_THRESHOLD")
    [ -n "$NUM_TOP_QUERIES" ]   && WSI_ARGS+=(--num-top-queries "$NUM_TOP_QUERIES")
    [ "$SAVE_JSON" = "True" ]   || WSI_ARGS+=(--no-json)
    [ -n "$CENTER_MODE" ]       && WSI_ARGS+=(-u "LSPDetrDetection.center_mode=$CENTER_MODE")

    cd "$DET_DIR"
    "$PY" -u "$WSI_SCRIPT" "${WSI_ARGS[@]}" $EXTRA_ARGS "$@" 2>&1 | tee "$LOG_FILE"
    exit_code=${PIPESTATUS[0]}
    if [ $exit_code -ne 0 ]; then
        echo "Error: WSI inference failed (exit code: $exit_code) - see $LOG_FILE"
        exit $exit_code
    fi
    echo "Completed. Results in $OUTPUT_DIR"
    exit 0
fi

# ---------------------------------------------------------------------------- run (images)
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
