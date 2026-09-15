# Vendored Dome-DETR runtime

`det/` is a port of LSP-DETR onto the Dome-DETR training/eval framework. So that a bare clone of this
branch trains without any sibling repository, the parts of that framework the port imports are copied here
and `det/lsp_det/__init__.py` puts this directory on `sys.path` (an explicit `$DOME_ROOT` still overrides it).

| item | source | notes |
|---|---|---|
| `src/` (91 modules) | `AIVIS-DETECTION/AIVIS-Dome-DETR/src/` | whole package, `__pycache__` stripped; `src/zoo/dome/ops/` (C++/CUDA MSDeformAttn sources) is dead code - nothing imports `deformable_encoder.py` - and is never built |
| `tools/visualize_src_flatten.py`, `tools/visualize_image_annotation.py`, `tools/concatenate_images.py` | `.../AIVIS-Dome-DETR/tools/` | the only `tools.*` modules `src/` imports (at import time, from `hybrid_encoder.py` / `dome_decoder.py` / `get_roi_features.py` / `solver/det_engine.py`); namespace package, no `__init__.py` as upstream. Checked by parsing every import of the vendored code (AST) and by importing every `src.*` submodule from a bare clone |
| `configs/` | `.../AIVIS-Dome-DETR/configs/` | `runtime.yml` + `dome/include/{dataloader,optimizer}.yml` are what `det/configs/LSP-T-*.yml` include; the rest is kept for `scripts/cfg_diff_vs_dome.py`-style comparisons (their dataset paths are the Dome box's) |
| `LICENSE` | `.../AIVIS-Dome-DETR/LICENSE` | Apache License 2.0 (D-FINE / Dome-DETR) - applies to everything in this directory |

## Provenance
* Copied 2026-09-15 from `/home/work/tksong/AIVIS-DETECTION` (single commit `1e6a557ea676640cbe83478f22c97ef33398bbf4`)
  **working tree**, i.e. including the one code change that repo never committed:
  `src/zoo/dome/dome_criterion.py` solves the per-layer Hungarian matches on a thread pool
  (`_get_match_pool()`, `DOME_MATCH_THREADS`, default 6; `<=1` = original sequential loop). It is
  result-identical to the sequential version and is what every det/ run (P4 TNT, HER2) trained with;
  `det/lsp_det/lsp_criterion.py` imports `_get_match_pool` from it (with a sequential fallback should
  `$DOME_ROOT` point at a pristine checkout).
  `configs/dataset/coco_detection.yml` also carries that tree's uncommitted path edits (unused by det/).
* Verified against the source tree with `diff -r` at copy time (no local edits on top).

## Re-syncing
```bash
DOME=/home/work/tksong/AIVIS-DETECTION/AIVIS-Dome-DETR
rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' "$DOME/src/"     det/third_party/dome/src/
rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' "$DOME/configs/" det/third_party/dome/configs/
cp "$DOME"/tools/{visualize_src_flatten,visualize_image_annotation,concatenate_images}.py det/third_party/dome/tools/
cp "$DOME/LICENSE" det/third_party/dome/LICENSE
```
Then run `det/tests` and a smoke (`det/README.md`) and record the source commit / diff here. Do **not** edit files
under this directory directly: the whole point of the port is that Dome behaviour stays byte-identical to the
baselines it is compared against.
