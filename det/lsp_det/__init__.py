"""LSP-DETR detection port living *outside* the Dome repo.

Importing this package
  1. puts the Dome repo root (``$DOME_ROOT`` or the sibling
     ``AIVIS-DETECTION/AIVIS-Dome-DETR`` checkout) on ``sys.path`` so ``src.*`` / ``tools.*`` import,
  2. imports ``src.zoo`` (Dome registrations) and then the ``lsp_det`` modules, which register
     ``LSPDetrDetection``, ``LSPCriterion``, ``RandomCropWithGridNoDummy``, ``ImageNetNormalize`` in
     the same registry.
Dome files are never modified.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
DET_ROOT = os.path.dirname(_HERE)                       # .../lsp-detr/det
LSP_REPO_ROOT = os.path.dirname(DET_ROOT)               # .../lsp-detr
DEFAULT_DOME_ROOT = os.path.abspath(os.path.join(LSP_REPO_ROOT, "..", "AIVIS-DETECTION", "AIVIS-Dome-DETR"))
DOME_ROOT = os.environ.get("DOME_ROOT", DEFAULT_DOME_ROOT)
HF5CLASS_CKPT = os.path.join(LSP_REPO_ROOT, "hf-5class", "model.safetensors")


def bootstrap_dome(dome_root: str = DOME_ROOT) -> str:
    """Make the Dome repo importable as ``src`` / ``tools`` (idempotent)."""
    if not os.path.isdir(os.path.join(dome_root, "src", "core")):
        raise ImportError(f"Dome repo not found at {dome_root!r}; set $DOME_ROOT")
    if dome_root not in sys.path:
        sys.path.insert(0, dome_root)
    # Dome's own modules must win over anything else called `src`
    import src.core  # noqa: F401
    import src.zoo   # noqa: F401  (registers DOME / DomeCriterion / HungarianMatcher / DomePostProcessor ...)
    import src.data  # noqa: F401  (registers datasets / transforms / collate)
    import src.optim  # noqa: F401 (ModelEMA, optimizers, schedulers)
    return dome_root


bootstrap_dome()

# register the LSP arm components (order: transforms first, they only depend on src.data)
from . import transforms as transforms  # noqa: E402,F401
from . import lsp_trunk as lsp_trunk  # noqa: E402,F401
from . import checkpoint as checkpoint  # noqa: E402,F401
from . import lsp_detr_det as lsp_detr_det  # noqa: E402,F401
try:  # training-only component
    from . import lsp_criterion as lsp_criterion  # noqa: E402,F401
except ImportError as _e:  # pragma: no cover - depends on the $DOME_ROOT checkout
    # ``lsp_criterion`` needs ``_get_match_pool`` from Dome's dome_criterion; older Dome
    # checkouts do not have it. Nothing on the inference path (det/inference.py,
    # det/wsi_infer.py) builds a criterion, so keep the package importable and let
    # det/train.py raise on ``CRITERION_IMPORT_ERROR`` instead.
    lsp_criterion = None  # type: ignore[assignment]
    CRITERION_IMPORT_ERROR: "Exception | None" = _e
else:
    CRITERION_IMPORT_ERROR = None
from . import optim_audit as optim_audit  # noqa: E402,F401

__all__ = ["DOME_ROOT", "DET_ROOT", "LSP_REPO_ROOT", "HF5CLASS_CKPT", "bootstrap_dome",
           "CRITERION_IMPORT_ERROR"]
