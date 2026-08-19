"""Transforms registered for the LSP arm (Dome's ``_transforms.py`` is not modified).

* ``RandomCropWithGridNoDummy`` (§6.1): identical to Dome's ``RandomCropWithGrid`` except that an
  empty crop stays empty (``boxes`` shape [0,4], ``labels`` shape [0]) instead of receiving a fake
  3x3 class-0 box. Implemented by overriding the single ``_add_dummy_box`` hook that all three
  call sites (single-crop / small-image / large-image paths) go through.
* ``ImageNetNormalize`` (§6.2): torchvision v2 ``Normalize`` with the ImageNet mean/std used by the
  hf-5class preprocessor, applied exactly once (after ``ConvertPILImage`` /255). It marks the
  image tensor it produced so a second application raises instead of silently double-normalising.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torchvision.transforms.v2 as T
from torchvision import tv_tensors

from src.core import register
from src.data.transforms._transforms import RandomCropWithGrid

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

__all__ = ["RandomCropWithGridNoDummy", "ImageNetNormalize", "IMAGENET_MEAN", "IMAGENET_STD"]


@register()
class RandomCropWithGridNoDummy(RandomCropWithGrid):
    """RandomCropWithGrid without the fake 3x3 class-0 GT on empty crops."""

    def __init__(self, crop_size: int, crops_per_call: int = 1, center_gt_size=None) -> None:
        super().__init__(crop_size=crop_size, crops_per_call=crops_per_call, center_gt_size=center_gt_size)

    def _add_dummy_box(self, image, target):
        # keep [0,4] boxes / [0] labels exactly as produced by _crop_boxes
        return target


class _DoubleNormalizeError(RuntimeError):
    pass


@register()
class ImageNetNormalize(T.Normalize):
    """ImageNet mean/std normalisation, guarded against double application."""

    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD, inplace: bool = False) -> None:
        super().__init__(mean=list(mean), std=list(std), inplace=inplace)

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        is_image = isinstance(inpt, tv_tensors.Image) or (type(inpt) is torch.Tensor)
        if not is_image:
            # BoundingBoxes etc.: torchvision passes them through (no normalize kernel) - keep that.
            return super().transform(inpt, params)
        if getattr(inpt, "_lsp_imagenet_normalized", False):
            raise _DoubleNormalizeError("ImageNetNormalize applied twice to the same image tensor (§6.2)")
        if torch.is_floating_point(inpt) and inpt.numel() > 0:
            with torch.no_grad():
                lo, hi = float(inpt.min()), float(inpt.max())
            if lo < -0.5 or hi > 1.5:
                raise _DoubleNormalizeError(
                    f"ImageNetNormalize input range [{lo:.3f},{hi:.3f}] is not [0,1]; "
                    "expected ConvertPILImage(/255) output exactly once before this op")
        out = super().transform(inpt, params)
        try:
            out._lsp_imagenet_normalized = True
        except Exception:
            pass
        return out
