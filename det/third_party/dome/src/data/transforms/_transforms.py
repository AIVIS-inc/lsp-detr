"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

from typing import Any, Dict, List, Optional

import PIL
import PIL.Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

from ...core import register
from .._misc import (
    BoundingBoxes,
    Image,
    Mask,
    SanitizeBoundingBoxes,
    Video,
    _boxes_keys,
    convert_to_tv_tensor,
)

torchvision.disable_beta_transforms_warning()


ColorJitter = register()(T.ColorJitter)
RandomVerticalFlip = register()(T.RandomVerticalFlip)
RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
Resize = register()(T.Resize)
# ToImageTensor = register()(T.ToImageTensor)
# ConvertDtype = register()(T.ConvertDtype)
# PILToTensor = register()(T.PILToTensor)
SanitizeBoundingBoxes = register(name="SanitizeBoundingBoxes")(SanitizeBoundingBoxes)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)


@register()
class EmptyTransform(T.Transform):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register()
class PadToSize(T.Pad):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sp = F.get_spatial_size(flat_inputs[0])
        h, w = self.size[1] - sp[0], self.size[0] - sp[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, size, fill=0, padding_mode="constant") -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = self._fill[type(inpt)]
        padding = params["padding"]
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]["padding"] = torch.tensor(self.padding)
        return outputs


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2,
        sampler_options: Optional[List[float]] = None,
        trials: int = 40,
        p: float = 1.0,
    ):
        super().__init__(
            min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials
        )
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (BoundingBoxes,)

    def __init__(self, fmt="", normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(
                inpt, key="boxes", box_format=self.fmt.upper(), spatial_size=spatial_size
            )

        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt


@register()
class RandomGaussianBlur(nn.Module):
    """Random Gaussian Blur with probability p"""
    
    def __init__(self, kernel_size, sigma=[0.1, 2.0], p=0.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p
        # Pre-create the GaussianBlur transform
        self.gaussian_blur = T.GaussianBlur(kernel_size=self.kernel_size, sigma=self.sigma)
        
    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        
        # Apply blur with probability p
        if torch.rand(1) >= self.p:
            return inputs
        
        # Apply GaussianBlur to appropriate inputs
        if len(inputs) > 1:
            # Multiple inputs case (image + target + ...)
            result = []
            for i, inp in enumerate(inputs):
                if i == 0:  # First input is typically the image
                    try:
                        result.append(self.gaussian_blur(inp))
                    except:
                        # If blur fails, keep original
                        result.append(inp)
                else:
                    # Keep other inputs unchanged (targets, etc.)
                    result.append(inp)
            return tuple(result)
        else:
            # Single input case
            try:
                return self.gaussian_blur(inputs)
            except:
                # If blur fails, keep original
                return inputs


@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (PIL.Image.Image,)

    def __init__(self, dtype="float32", scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == "float32":
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.0

        inpt = Image(inpt)

        return inpt
@register()
class RandomCropWithGrid(T.Transform):
    """
    Random crop with CENTER GT region approach (matches crop_coco_center_gt.py logic).
    
    NEW LOGIC:
    - Divides image into non-overlapping center_gt_size tiles (stride = center_gt_size)
    - Each tile is placed at the CENTER of a crop_size crop
    - Context is added around each tile: (crop_size - center_gt_size) / 2 on each side
    - Reflection padding is applied for boundary crops
    
    Example (4096 image, crop_size=2048, center_gt_size=512):
    - stride = center_gt_size = 512
    - context_margin = (2048 - 512) / 2 = 768
    - Center GT tiles: [0,512], [512,1024], ..., [3584,4096]
    - Number of crops: ceil(4096/512) = 8×8 = 64 crops
    - Tile [0,512] → crop [-768, 1280] (with padding L=768, T=768)
    - In cropped image, tile is always at [768, 768] to [1280, 1280] (center)
    
    - For small images (width < crop_size OR height < crop_size):
      Applies reflection padding and crops from center.
    
    - For large images:
      1. Generates grid coordinates (stride = center_gt_size)
      2. Randomly selects a grid point OR uses specified grid_index from target
      3. Crops with center GT region centered
      4. Applies reflection padding for boundary crops
      5. Filters/clips bounding boxes based on center coordinates
      6. If no bounding boxes remain, adds a dummy 3x3 box in center GT region
    
    Multi-Crop Support:
    - If target contains 'grid_index', uses that specific grid location
    - Otherwise, randomly selects a grid location
    - Works seamlessly with MultiCropDatasetWrapper
    
    Args:
        crop_size (int): Final output size (square).
        crops_per_call (int): Number of crops to generate per call (default: 1).
                             If > 1, returns list of (image, target) tuples.
                             Note: For dataset integration, use MultiCropDatasetWrapper instead.
        center_gt_size (int, optional): Size of center GT region (also acts as stride).
                                       If None, defaults to crop_size // 4.
    """
    
    def __init__(
        self,
        crop_size: int,
        crops_per_call: int = 1,
        center_gt_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.crop_size = crop_size
        # NEW: stride = center_gt_size (default: crop_size // 4)
        # This matches the logic in crop_coco_center_gt.py
        self.center_gt_size = center_gt_size if center_gt_size is not None else (crop_size // 4)
        self.stride = self.center_gt_size  # stride = center_gt_size
        self.context_margin = (crop_size - self.center_gt_size) // 2
        self.crops_per_call = crops_per_call
        
    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        
        # Handle both (image, target) and single input cases
        if isinstance(inputs, (tuple, list)) and len(inputs) >= 2:
            image = inputs[0]
            target = inputs[1]
            extra_inputs = inputs[2:] if len(inputs) > 2 else ()
        else:
            # Single input case (no target)
            image = inputs
            target = None
            extra_inputs = ()
        
        # Multi-crop mode: generate multiple crops per call
        if self.crops_per_call > 1:
            return self._multi_crop_forward(image, target, extra_inputs)
        
        # Single crop mode (default)
        return self._single_crop_forward(image, target, extra_inputs)
    
    def _single_crop_forward(self, image, target, extra_inputs):
        """Process single crop (default behavior)"""
        # Get image size (optimized)
        if isinstance(image, PIL.Image.Image):
            img_w, img_h = image.size
        elif isinstance(image, torch.Tensor):
            img_h, img_w = image.shape[-2:]  # Works for both 3D and 4D tensors
        elif hasattr(image, 'spatial_size'):
            img_h, img_w = image.spatial_size
        else:
            # Cannot determine image size, return as is
            if target is not None:
                if extra_inputs:
                    return (image, target) + extra_inputs
                return image, target
            return image
        
        # Extract grid_index from target if provided (for MultiCropDatasetWrapper)
        grid_index = None
        if target is not None and 'grid_index' in target:
            grid_index = target.pop('grid_index')  # Remove from target after reading
        
        # Case 1: Small image - apply padding and center crop
        if img_w <= self.crop_size or img_h <= self.crop_size:
            image, target = self._handle_small_image(image, target, img_w, img_h, grid_index)
        else:
            # Case 2: Large image - grid-based random crop with multi-scale
            image, target = self._handle_large_image(image, target, img_w, img_h, grid_index)
        
        # Add dummy box if no bounding boxes remain
        if target is not None and "boxes" in target:
            if len(target["boxes"]) == 0:
                target = self._add_dummy_box(image, target)
        
        # Return in the same format as input
        if target is not None:
            if extra_inputs:
                return (image, target) + extra_inputs
            else:
                return image, target
        else:
            return image
    
    def _multi_crop_forward(self, image, target, extra_inputs):
        """
        Generate multiple crops per call.
        
        Returns:
            List of (image, target) tuples
        
        Note: This mode is primarily for special use cases.
        For standard training, use MultiCropDatasetWrapper instead.
        """
        results = []
        
        for crop_idx in range(self.crops_per_call):
            # Deep copy target to avoid modifying original
            import copy
            target_copy = copy.deepcopy(target) if target is not None else None
            
            # Add crop index to target for tracking
            if target_copy is not None:
                target_copy['crop_idx'] = crop_idx
            
            # Process single crop
            crop_result = self._single_crop_forward(image, target_copy, extra_inputs)
            results.append(crop_result)
        
        return results
    
    def _handle_small_image(self, image, target, img_w, img_h, grid_index=None):
        """
        Handle images smaller than crop_size with reflection padding.
        
        SIMPLIFIED LOGIC:
        1. Calculate symmetric padding to center the image
        2. Apply reflection padding to image
        3. Reflect GT boxes to match image reflection padding
        4. Apply center crop to crop_size
        5. Filter boxes based on crop region (using _crop_boxes)
        
        This approach is mathematically correct and easy to understand.
        """
        orig_w, orig_h = img_w, img_h
        
        # Step 1: Calculate padding (symmetric to center the image)
        pad_w = max(0, self.crop_size - img_w)
        pad_h = max(0, self.crop_size - img_h)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        
        # Step 2: Extract and convert boxes to XYXY format
        original_boxes_xyxy = None
        original_labels = None
        original_area = None
        original_iscrowd = None
        original_box_format = None
        
        if target is not None and "boxes" in target and len(target["boxes"]) > 0:
            boxes = target["boxes"]
            
            if isinstance(boxes, BoundingBoxes):
                box_data = boxes.data.clone()
                original_box_format = boxes.format
            else:
                box_data = boxes.clone()
                original_box_format = "XYXY"
            
            # Convert to XYXY for reflection calculations
            if original_box_format == "XYWH":
                x1 = box_data[:, 0]
                y1 = box_data[:, 1]
                x2 = box_data[:, 0] + box_data[:, 2]
                y2 = box_data[:, 1] + box_data[:, 3]
                original_boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
            elif original_box_format == "CXCYWH":
                x1 = box_data[:, 0] - box_data[:, 2] / 2
                y1 = box_data[:, 1] - box_data[:, 3] / 2
                x2 = box_data[:, 0] + box_data[:, 2] / 2
                y2 = box_data[:, 1] + box_data[:, 3] / 2
                original_boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
            else:  # XYXY
                original_boxes_xyxy = box_data.clone()
            
            # Store metadata
            if "labels" in target:
                original_labels = target["labels"].clone()
            if "area" in target:
                original_area = target["area"].clone()
            if "iscrowd" in target:
                original_iscrowd = target["iscrowd"].clone()
        
        # Step 3: Apply reflection padding to image
        if pad_w > 0 or pad_h > 0:
            # Convert to tensor for padding
            if isinstance(image, PIL.Image.Image):
                img_tensor = F.pil_to_tensor(image).float()
            elif isinstance(image, torch.Tensor):
                img_tensor = image.float()
            elif isinstance(image, Image):
                img_tensor = image.float()
            else:
                img_tensor = F.pil_to_tensor(image).float()
            
            # Check if direct padding is possible
            can_use_reflect_w = (pad_left < img_w and pad_right < img_w)
            can_use_reflect_h = (pad_top < img_h and pad_bottom < img_h)
            
            # Track total growth for box coordinate adjustment
            total_grow_left = 0
            total_grow_top = 0
            
            # For very small images, grow iteratively first
            if not can_use_reflect_w:
                # Grow horizontally until we can apply final padding
                target_width = max(pad_left + 1, pad_right + 1)
                while img_w < target_width:
                    grow = min(img_w - 1, (target_width - img_w + 1) // 2)
                    if grow <= 0:
                        break
                    img_tensor = torch.nn.functional.pad(img_tensor, (grow, grow, 0, 0), mode='reflect')
                    total_grow_left += grow
                    img_w = img_tensor.shape[-1]
                
                # Update boxes for horizontal growth
                if original_boxes_xyxy is not None and total_grow_left > 0:
                    grow_offset_x = torch.tensor(
                        [total_grow_left, 0, total_grow_left, 0],
                        dtype=original_boxes_xyxy.dtype,
                        device=original_boxes_xyxy.device
                    )
                    original_boxes_xyxy = original_boxes_xyxy + grow_offset_x
                
                # Recalculate padding
                pad_w = max(0, self.crop_size - img_w)
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
            
            if not can_use_reflect_h:
                # Grow vertically until we can apply final padding
                target_height = max(pad_top + 1, pad_bottom + 1)
                while img_h < target_height:
                    grow = min(img_h - 1, (target_height - img_h + 1) // 2)
                    if grow <= 0:
                        break
                    img_tensor = torch.nn.functional.pad(img_tensor, (0, 0, grow, grow), mode='reflect')
                    total_grow_top += grow
                    img_h = img_tensor.shape[-2]
                
                # Update boxes for vertical growth
                if original_boxes_xyxy is not None and total_grow_top > 0:
                    grow_offset_y = torch.tensor(
                        [0, total_grow_top, 0, total_grow_top],
                        dtype=original_boxes_xyxy.dtype,
                        device=original_boxes_xyxy.device
                    )
                    original_boxes_xyxy = original_boxes_xyxy + grow_offset_y
                
                # Recalculate padding
                pad_h = max(0, self.crop_size - img_h)
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
            
            # Update reflection axis to grown image size (before final padding)
            # This is critical: reflection should be relative to the grown image, not original
            grown_w = img_w  # Size after iterative growing, before final padding
            grown_h = img_h
            
            # Apply final padding
            if pad_w > 0 or pad_h > 0:
                img_tensor = torch.nn.functional.pad(
                    img_tensor,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode='reflect'
                )
            
            # Convert back to original format
            img_tensor = img_tensor.byte()
            if isinstance(image, PIL.Image.Image):
                image = F.to_pil_image(img_tensor)
                img_w, img_h = image.size
            elif isinstance(image, torch.Tensor):
                image = img_tensor
                img_h, img_w = image.shape[-2:]
            elif isinstance(image, Image):
                image = Image(img_tensor)
                img_h, img_w = image.shape[-2:]
            
            # Step 4: Reflect GT boxes to match image padding
            # CRITICAL FIX: Reflection axis = ORIGINAL IMAGE BOUNDARY, not (0,0)!
            # 
            # After all padding, in final 2048 image:
            #   Original 512x512 image is at [total_grow + pad, total_grow + pad + 512]
            #   For our example: [129+639, 129+639+512] = [768, 1280]
            #
            # Reflection must mirror around these boundaries (768, 1280)
            # But we're working in grown coordinate system before offset
            # Reflection axis in grown coords = total_grow (where original image starts)
            #
            if original_boxes_xyxy is not None and len(original_boxes_xyxy) > 0:
                all_boxes = [original_boxes_xyxy]
                all_labels = [original_labels] if original_labels is not None else []
                all_area = [original_area] if original_area is not None else []
                all_iscrowd = [original_iscrowd] if original_iscrowd is not None else []
                
                # Reflection axes in grown coordinate system (before final offset)
                # Original image in grown coords: [total_grow_left, total_grow_left + orig_w]
                # Reflection mirrors around these boundaries
                axis_left = total_grow_left  # Left boundary of original image
                axis_right = total_grow_left + orig_w  # Right boundary
                axis_top = total_grow_top
                axis_bottom = total_grow_top + orig_h
                
                # Reflect for each padding direction
                if pad_left > 0:
                    # Left reflection: mirror around x = axis_left
                    # x' = 2*axis_left - x
                    # Only reflect boxes near left boundary: x < axis_left + pad_left
                    mask = original_boxes_xyxy[:, 0] < (axis_left + pad_left)
                    if mask.any():
                        reflected = original_boxes_xyxy[mask].clone()
                        # Mirror around axis_left
                        reflected[:, 0] = 2 * axis_left - original_boxes_xyxy[mask, 2]
                        reflected[:, 2] = 2 * axis_left - original_boxes_xyxy[mask, 0]
                        all_boxes.append(reflected)
                        if original_labels is not None:
                            all_labels.append(original_labels[mask].clone())
                        if original_area is not None:
                            all_area.append(original_area[mask].clone())
                        if original_iscrowd is not None:
                            all_iscrowd.append(original_iscrowd[mask].clone())
                
                if pad_right > 0:
                    # Right reflection: mirror around x = axis_right
                    # x' = 2*axis_right - x
                    mask = original_boxes_xyxy[:, 2] > (axis_right - pad_right)
                    if mask.any():
                        reflected = original_boxes_xyxy[mask].clone()
                        reflected[:, 0] = 2 * axis_right - original_boxes_xyxy[mask, 2]
                        reflected[:, 2] = 2 * axis_right - original_boxes_xyxy[mask, 0]
                        all_boxes.append(reflected)
                        if original_labels is not None:
                            all_labels.append(original_labels[mask].clone())
                        if original_area is not None:
                            all_area.append(original_area[mask].clone())
                        if original_iscrowd is not None:
                            all_iscrowd.append(original_iscrowd[mask].clone())
                
                if pad_top > 0:
                    # Top reflection: mirror around y = axis_top
                    # y' = 2*axis_top - y
                    mask = original_boxes_xyxy[:, 1] < (axis_top + pad_top)
                    if mask.any():
                        reflected = original_boxes_xyxy[mask].clone()
                        reflected[:, 1] = 2 * axis_top - original_boxes_xyxy[mask, 3]
                        reflected[:, 3] = 2 * axis_top - original_boxes_xyxy[mask, 1]
                        all_boxes.append(reflected)
                        if original_labels is not None:
                            all_labels.append(original_labels[mask].clone())
                        if original_area is not None:
                            all_area.append(original_area[mask].clone())
                        if original_iscrowd is not None:
                            all_iscrowd.append(original_iscrowd[mask].clone())
                
                if pad_bottom > 0:
                    # Bottom reflection: mirror around y = axis_bottom
                    # y' = 2*axis_bottom - y
                    mask = original_boxes_xyxy[:, 3] > (axis_bottom - pad_bottom)
                    if mask.any():
                        reflected = original_boxes_xyxy[mask].clone()
                        reflected[:, 1] = 2 * axis_bottom - original_boxes_xyxy[mask, 3]
                        reflected[:, 3] = 2 * axis_bottom - original_boxes_xyxy[mask, 1]
                        all_boxes.append(reflected)
                        if original_labels is not None:
                            all_labels.append(original_labels[mask].clone())
                        if original_area is not None:
                            all_area.append(original_area[mask].clone())
                        if original_iscrowd is not None:
                            all_iscrowd.append(original_iscrowd[mask].clone())
                
                # Corner reflections (diagonal) - mirror around both axes
                if pad_left > 0 and pad_top > 0:
                    # Top-left corner
                    mask = (original_boxes_xyxy[:, 0] < (axis_left + pad_left)) & (original_boxes_xyxy[:, 1] < (axis_top + pad_top))
                    if mask.any():
                        reflected = original_boxes_xyxy[mask].clone()
                        reflected[:, 0] = 2 * axis_left - original_boxes_xyxy[mask, 2]
                        reflected[:, 2] = 2 * axis_left - original_boxes_xyxy[mask, 0]
                        reflected[:, 1] = 2 * axis_top - original_boxes_xyxy[mask, 3]
                        reflected[:, 3] = 2 * axis_top - original_boxes_xyxy[mask, 1]
                        all_boxes.append(reflected)
                        if original_labels is not None:
                            all_labels.append(original_labels[mask].clone())
                        if original_area is not None:
                            all_area.append(original_area[mask].clone())
                        if original_iscrowd is not None:
                            all_iscrowd.append(original_iscrowd[mask].clone())
                
                if pad_right > 0 and pad_top > 0:
                    # Top-right corner
                    mask = (original_boxes_xyxy[:, 2] > (axis_right - pad_right)) & (original_boxes_xyxy[:, 1] < (axis_top + pad_top))
                    if mask.any():
                        reflected = original_boxes_xyxy[mask].clone()
                        reflected[:, 0] = 2 * axis_right - original_boxes_xyxy[mask, 2]
                        reflected[:, 2] = 2 * axis_right - original_boxes_xyxy[mask, 0]
                        reflected[:, 1] = 2 * axis_top - original_boxes_xyxy[mask, 3]
                        reflected[:, 3] = 2 * axis_top - original_boxes_xyxy[mask, 1]
                        all_boxes.append(reflected)
                        if original_labels is not None:
                            all_labels.append(original_labels[mask].clone())
                        if original_area is not None:
                            all_area.append(original_area[mask].clone())
                        if original_iscrowd is not None:
                            all_iscrowd.append(original_iscrowd[mask].clone())
                
                if pad_left > 0 and pad_bottom > 0:
                    # Bottom-left corner
                    mask = (original_boxes_xyxy[:, 0] < (axis_left + pad_left)) & (original_boxes_xyxy[:, 3] > (axis_bottom - pad_bottom))
                    if mask.any():
                        reflected = original_boxes_xyxy[mask].clone()
                        reflected[:, 0] = 2 * axis_left - original_boxes_xyxy[mask, 2]
                        reflected[:, 2] = 2 * axis_left - original_boxes_xyxy[mask, 0]
                        reflected[:, 1] = 2 * axis_bottom - original_boxes_xyxy[mask, 3]
                        reflected[:, 3] = 2 * axis_bottom - original_boxes_xyxy[mask, 1]
                        all_boxes.append(reflected)
                        if original_labels is not None:
                            all_labels.append(original_labels[mask].clone())
                        if original_area is not None:
                            all_area.append(original_area[mask].clone())
                        if original_iscrowd is not None:
                            all_iscrowd.append(original_iscrowd[mask].clone())
                
                if pad_right > 0 and pad_bottom > 0:
                    # Bottom-right corner
                    mask = (original_boxes_xyxy[:, 2] > (axis_right - pad_right)) & (original_boxes_xyxy[:, 3] > (axis_bottom - pad_bottom))
                    if mask.any():
                        reflected = original_boxes_xyxy[mask].clone()
                        reflected[:, 0] = 2 * axis_right - original_boxes_xyxy[mask, 2]
                        reflected[:, 2] = 2 * axis_right - original_boxes_xyxy[mask, 0]
                        reflected[:, 1] = 2 * axis_bottom - original_boxes_xyxy[mask, 3]
                        reflected[:, 3] = 2 * axis_bottom - original_boxes_xyxy[mask, 1]
                        all_boxes.append(reflected)
                        if original_labels is not None:
                            all_labels.append(original_labels[mask].clone())
                        if original_area is not None:
                            all_area.append(original_area[mask].clone())
                        if original_iscrowd is not None:
                            all_iscrowd.append(original_iscrowd[mask].clone())
                
                # Concatenate all reflected boxes
                original_boxes_xyxy = torch.cat(all_boxes, dim=0)
                if len(all_labels) > 0:
                    original_labels = torch.cat(all_labels, dim=0)
                if len(all_area) > 0:
                    original_area = torch.cat(all_area, dim=0)
                if len(all_iscrowd) > 0:
                    original_iscrowd = torch.cat(all_iscrowd, dim=0)
                
                # Shift all boxes by padding offset
                offset = torch.tensor(
                    [pad_left, pad_top, pad_left, pad_top],
                    dtype=original_boxes_xyxy.dtype,
                    device=original_boxes_xyxy.device
                )
                original_boxes_xyxy = original_boxes_xyxy + offset
                
                # Convert back to original format and update target
                boxes = target["boxes"]
                if isinstance(boxes, BoundingBoxes):
                    box_fmt = boxes.format.value.lower()
                    if box_fmt == "xywh":
                        w = original_boxes_xyxy[:, 2] - original_boxes_xyxy[:, 0]
                        h = original_boxes_xyxy[:, 3] - original_boxes_xyxy[:, 1]
                        original_boxes_xyxy = torch.stack([
                            original_boxes_xyxy[:, 0],
                            original_boxes_xyxy[:, 1],
                            w, h
                        ], dim=1)
                    elif box_fmt == "cxcywh":
                        w = original_boxes_xyxy[:, 2] - original_boxes_xyxy[:, 0]
                        h = original_boxes_xyxy[:, 3] - original_boxes_xyxy[:, 1]
                        cx = original_boxes_xyxy[:, 0] + w / 2
                        cy = original_boxes_xyxy[:, 1] + h / 2
                        original_boxes_xyxy = torch.stack([cx, cy, w, h], dim=1)
                    
                    boxes_kwargs = {"format": boxes.format}
                    if _boxes_keys[1] == "canvas_size":
                        boxes_kwargs["canvas_size"] = (img_h, img_w)
                    else:
                        boxes_kwargs["spatial_size"] = (img_h, img_w)
                    target["boxes"] = BoundingBoxes(original_boxes_xyxy, **boxes_kwargs)
                else:
                    target["boxes"] = original_boxes_xyxy
                
                # Update metadata
                if original_labels is not None:
                    target["labels"] = original_labels
                if original_area is not None:
                    target["area"] = original_area
                if original_iscrowd is not None:
                    target["iscrowd"] = original_iscrowd
        
        # Step 5: Center crop to crop_size
        crop_left = (img_w - self.crop_size) // 2
        crop_top = (img_h - self.crop_size) // 2
        crop_right = crop_left + self.crop_size
        crop_bottom = crop_top + self.crop_size
        
        # Crop image
        if isinstance(image, PIL.Image.Image):
            image = image.crop((crop_left, crop_top, crop_right, crop_bottom))
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image[:, crop_top:crop_bottom, crop_left:crop_right]
            elif image.dim() == 4:
                image = image[:, :, crop_top:crop_bottom, crop_left:crop_right]
        elif isinstance(image, Image):
            image = image[:, crop_top:crop_bottom, crop_left:crop_right]
        
        # Step 6: Filter boxes based on crop region
        if target is not None and "boxes" in target:
            target = self._crop_boxes(
                target, crop_left, crop_top, crop_right, crop_bottom, self.center_gt_size
            )
            
            # Add dummy box if all boxes were filtered out
            if len(target["boxes"]) == 0:
                target = self._add_dummy_box(image, target)
        
        return image, target
    
    def _handle_large_image(self, image, target, img_w, img_h, grid_index=None):
        """
        Handle large images with CENTER GT region approach (matches crop_coco_center_gt.py).
        
        NEW LOGIC:
        1. Divide image into non-overlapping center_gt_size tiles (stride = center_gt_size)
        2. For each tile, place it at the CENTER of a crop_size crop
        3. Add context around the center tile: context_margin on each side
        4. Apply reflection padding if crop extends beyond image boundaries
        5. BOUNDARY HANDLING: Last tile is adjusted to stay within image bounds, 
           creating overlap with the previous tile if necessary
        
        For 4096 image with crop_size=2048, center_gt_size=512:
        - stride = center_gt_size = 512
        - context_margin = (2048 - 512) / 2 = 768
        - Center GT tiles: [0,512], [512,1024], [1024,1536], ..., [3584,4096]
        - Number of tiles: ceil(4096/512) = 8 per dimension → 8×8=64 crops
        - Tile [0,512] → crop [-768, 1280] (padding L=768, T=768)
        - Tile [3584,4096] → crop [2816, 4864] (padding R=768, B=768)
        
        For 4100 image (not divisible):
        - stride = 512, ceil(4100/512) = 9 tiles
        - Last tile adjusted: [4100-512, 4100] = [3588, 4100] instead of [4096, 4608]
        - This creates overlap with tile 7 [3584, 4096], but keeps center GT in real image
        
        Args:
            grid_index: If provided, use this specific grid location instead of random
        """
        # Use fixed crop_size (no multi-scale)
        scale = self.crop_size
        context_margin = self.context_margin
        
        # Calculate number of VALID center GT tiles (only tiles fully within image)
        # A tile at (ix, iy) is valid if: ix*stride + center_gt_size <= img_w
        max_valid_ix = max(0, int((img_w - self.center_gt_size) // self.stride))
        max_valid_iy = max(0, int((img_h - self.center_gt_size) // self.stride))
        
        # Number of valid tiles (0-indexed, so +1)
        num_tiles_x = max_valid_ix + 1
        num_tiles_y = max_valid_iy + 1
        
        if num_tiles_x <= 0 or num_tiles_y <= 0:
            # Image is too small, fall back to center crop
            center_x = 0
            center_y = 0
        else:
            # Select grid position: use grid_index if provided, otherwise random
            if grid_index is not None:
                # Convert linear grid_index to 2D coordinates
                total_grids = num_tiles_x * num_tiles_y
                grid_index = min(grid_index, total_grids - 1)  # Clamp to valid range
                ix = grid_index % num_tiles_x
                iy = grid_index // num_tiles_x
            else:
                # Randomly select a valid grid index
                ix = int(torch.randint(0, num_tiles_x, (1,)).item())
                iy = int(torch.randint(0, num_tiles_y, (1,)).item())
            
            # Calculate center GT tile position in original image
            # This is the top-left corner of the center GT region
            center_x = ix * self.stride
            center_y = iy * self.stride
            
            # No boundary check needed - num_tiles_x/y already exclude invalid grids
            # All selected grids are guaranteed to be valid (tile fully within image)
        
        # Calculate crop position (center GT region should be at the center of crop)
        # crop_x and crop_y can be negative (will need padding)
        crop_left = center_x - context_margin
        crop_top = center_y - context_margin
        crop_right = crop_left + scale
        crop_bottom = crop_top + scale
        
        # Check if padding is needed (boundary crosses image edges)
        needs_padding = (crop_left < 0 or crop_top < 0 or 
                        crop_right > img_w or crop_bottom > img_h)
        
        if needs_padding:
            # Apply reflection padding for boundary crops (2-stage if needed)
            pad_left = max(0, -crop_left)
            pad_top = max(0, -crop_top)
            pad_right = max(0, crop_right - img_w)
            pad_bottom = max(0, crop_bottom - img_h)
            
            # PyTorch reflection padding constraint: padding must be < input dimension
            can_use_reflect_w = (pad_left < img_w and pad_right < img_w)
            can_use_reflect_h = (pad_top < img_h and pad_bottom < img_h)
            
            # Convert to tensor for padding
            if isinstance(image, PIL.Image.Image):
                img_tensor = F.pil_to_tensor(image).float()
            elif isinstance(image, torch.Tensor):
                img_tensor = image.float()
            elif isinstance(image, Image):
                img_tensor = image.float()
            else:
                img_tensor = F.pil_to_tensor(image).float()
            
            # Stage 1: If padding > dimension, first grow image with reflection
            if not can_use_reflect_w or not can_use_reflect_h:
                if not can_use_reflect_w:
                    # Grow width
                    grow_pad_left = img_w - 1
                    grow_pad_right = img_w - 1
                    img_tensor = torch.nn.functional.pad(
                        img_tensor,
                        (grow_pad_left, grow_pad_right, 0, 0),
                        mode='reflect'
                    )
                    # Update boxes and recalculate padding
                    if target is not None and "boxes" in target:
                        boxes = target["boxes"]
                        offset = torch.tensor(
                            [grow_pad_left, 0, grow_pad_left, 0],
                            dtype=boxes.dtype if isinstance(boxes, torch.Tensor) else torch.float32,
                            device=boxes.device if isinstance(boxes, torch.Tensor) else 'cpu'
                        )
                        if isinstance(boxes, BoundingBoxes):
                            padded_w = img_tensor.shape[-1]
                            boxes_kwargs = {"format": boxes.format}
                            if _boxes_keys[1] == "canvas_size":
                                boxes_kwargs["canvas_size"] = (img_tensor.shape[-2], padded_w)
                            else:
                                boxes_kwargs["spatial_size"] = (img_tensor.shape[-2], padded_w)
                            boxes = BoundingBoxes(boxes.data + offset, **boxes_kwargs)
                        else:
                            boxes = boxes + offset
                        target["boxes"] = boxes
                    
                    # Update dimensions and recalculate padding
                    img_w = img_tensor.shape[-1]
                    crop_left += grow_pad_left
                    crop_right += grow_pad_left
                    pad_left = max(0, -crop_left)
                    pad_right = max(0, crop_right - img_w)
                
                if not can_use_reflect_h:
                    # Grow height
                    grow_pad_top = img_h - 1
                    grow_pad_bottom = img_h - 1
                    img_tensor = torch.nn.functional.pad(
                        img_tensor,
                        (0, 0, grow_pad_top, grow_pad_bottom),
                        mode='reflect'
                    )
                    # Update boxes and recalculate padding
                    if target is not None and "boxes" in target:
                        boxes = target["boxes"]
                        offset = torch.tensor(
                            [0, grow_pad_top, 0, grow_pad_top],
                            dtype=boxes.dtype if isinstance(boxes, torch.Tensor) else torch.float32,
                            device=boxes.device if isinstance(boxes, torch.Tensor) else 'cpu'
                        )
                        if isinstance(boxes, BoundingBoxes):
                            padded_h = img_tensor.shape[-2]
                            boxes_kwargs = {"format": boxes.format}
                            if _boxes_keys[1] == "canvas_size":
                                boxes_kwargs["canvas_size"] = (padded_h, img_tensor.shape[-1])
                            else:
                                boxes_kwargs["spatial_size"] = (padded_h, img_tensor.shape[-1])
                            boxes = BoundingBoxes(boxes.data + offset, **boxes_kwargs)
                        else:
                            boxes = boxes + offset
                        target["boxes"] = boxes
                    
                    # Update dimensions and recalculate padding
                    img_h = img_tensor.shape[-2]
                    crop_top += grow_pad_top
                    crop_bottom += grow_pad_top
                    pad_top = max(0, -crop_top)
                    pad_bottom = max(0, crop_bottom - img_h)
            
            # Stage 2: Apply final reflection padding
            if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
                img_tensor = torch.nn.functional.pad(
                    img_tensor,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode='reflect'
                )
            
            # Convert back to original format
            img_tensor = img_tensor.byte()
            if isinstance(image, PIL.Image.Image):
                image = F.to_pil_image(img_tensor)
            elif isinstance(image, torch.Tensor):
                image = img_tensor
            elif isinstance(image, Image):
                image = Image(img_tensor)
            
            # Adjust bounding boxes for padding offset FIRST
            if target is not None and "boxes" in target and len(target["boxes"]) > 0:
                boxes = target["boxes"]
                offset = torch.tensor(
                    [pad_left, pad_top, pad_left, pad_top],
                    dtype=boxes.dtype if isinstance(boxes, torch.Tensor) else torch.float32,
                    device=boxes.device if isinstance(boxes, torch.Tensor) else 'cpu'
                )
                
                if isinstance(boxes, BoundingBoxes):
                    # Update spatial size for padded image
                    padded_h = img_h + pad_top + pad_bottom
                    padded_w = img_w + pad_left + pad_right
                    
                    boxes_kwargs = {"format": boxes.format}
                    if _boxes_keys[1] == "canvas_size":
                        boxes_kwargs["canvas_size"] = (padded_h, padded_w)
                    else:
                        boxes_kwargs["spatial_size"] = (padded_h, padded_w)
                    
                    boxes = BoundingBoxes(boxes.data + offset, **boxes_kwargs)
                else:
                    boxes = boxes + offset
                
                target["boxes"] = boxes
            
            # Adjust crop coordinates after padding
            # Now crop coordinates are in the PADDED image coordinate system
            crop_left += pad_left
            crop_top += pad_top
            crop_right = crop_left + scale  # Recalculate based on new crop_left
            crop_bottom = crop_top + scale  # Recalculate based on new crop_top
        
        # Crop image at selected scale
        if isinstance(image, PIL.Image.Image):
            image = image.crop((crop_left, crop_top, crop_right, crop_bottom))
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image[:, crop_top:crop_bottom, crop_left:crop_right]
            elif image.dim() == 4:
                image = image[:, :, crop_top:crop_bottom, crop_left:crop_right]
        elif isinstance(image, Image):
            image = image[:, crop_top:crop_bottom, crop_left:crop_right]
        
        # Crop and filter bounding boxes
        if target is not None and "boxes" in target:
            # Use self.center_gt_size (no multi-scale, so no need to scale it)
            target = self._crop_boxes(
                target, crop_left, crop_top, crop_right, crop_bottom, self.center_gt_size
            )
            
            # Add dummy box if all boxes were filtered out
            if len(target["boxes"]) == 0:
                target = self._add_dummy_box(image, target)
        
        return image, target
    
    def _reflect_boundary(self, coord, min_val, max_val):
        """Apply reflection to coordinates that exceed boundaries."""
        if coord < min_val:
            coord = min_val + (min_val - coord)
        elif coord > max_val:
            coord = max_val - (coord - max_val)
        
        # Clamp to ensure within bounds after reflection
        coord = max(min_val, min(coord, max_val))
        return coord
    
    def _crop_boxes(self, target, crop_left, crop_top, crop_right, crop_bottom, center_gt_size_actual=None):
        """
        Crop and filter bounding boxes based on CENTER GT REGION.
        
        NEW LOGIC (matches crop_coco_center_gt.py):
        - Center GT region size = center_gt_size_actual (scaled for multi-scale)
        - Context margin = (crop_size - center_gt_size) / 2
        - Only keeps boxes whose center falls within the center GT region
        - Clips boxes that are partially inside
        
        For crop_size=2048, center_gt_size=512, scale=2048:
        - Center GT region: [768, 768] to [1280, 1280] in cropped image
        - Context: 768px margin on all sides
        
        Args:
            center_gt_size_actual: Actual center GT size after scaling (for multi-scale support)
                                  If None, uses self.center_gt_size
        """
        boxes = target["boxes"]
        
        # Get box format
        if isinstance(boxes, BoundingBoxes):
            box_fmt = boxes.format.value.lower()
        else:
            # Assume xyxy format
            box_fmt = "xyxy"
        
        # Convert to xyxy for processing (optimized: avoid zeros_like)
        if box_fmt == "xyxy":
            boxes_xyxy = boxes if not isinstance(boxes, BoundingBoxes) else boxes.data
        elif box_fmt == "xywh":
            # (x, y, w, h) -> (x1, y1, x2, y2)
            boxes_data = boxes if not isinstance(boxes, BoundingBoxes) else boxes.data
            boxes_xyxy = boxes_data.clone()
            boxes_xyxy[:, 2] = boxes_data[:, 0] + boxes_data[:, 2]  # x2 = x + w
            boxes_xyxy[:, 3] = boxes_data[:, 1] + boxes_data[:, 3]  # y2 = y + h
        elif box_fmt == "cxcywh":
            # (cx, cy, w, h) -> (x1, y1, x2, y2)
            boxes_data = boxes if not isinstance(boxes, BoundingBoxes) else boxes.data
            half_w = boxes_data[:, 2] / 2
            half_h = boxes_data[:, 3] / 2
            boxes_xyxy = boxes_data.clone()
            boxes_xyxy[:, 0] = boxes_data[:, 0] - half_w  # x1 = cx - w/2
            boxes_xyxy[:, 1] = boxes_data[:, 1] - half_h  # y1 = cy - h/2
            boxes_xyxy[:, 2] = boxes_data[:, 0] + half_w  # x2 = cx + w/2
            boxes_xyxy[:, 3] = boxes_data[:, 1] + half_h  # y2 = cy + h/2
        else:
            boxes_xyxy = boxes if not isinstance(boxes, BoundingBoxes) else boxes.data
        
        # Calculate box centers
        cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2
        cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2
        
        # Calculate CENTER GT REGION (based on actual scaled center_gt_size)
        crop_w = crop_right - crop_left
        crop_h = crop_bottom - crop_top
        
        # Use scaled center GT size if provided (for multi-scale)
        center_w = center_gt_size_actual if center_gt_size_actual is not None else self.center_gt_size
        center_h = center_w  # Assume square
        
        # Context margin = (crop_size - center_gt_size) / 2
        context_margin_w = (crop_w - center_w) // 2
        context_margin_h = (crop_h - center_h) // 2
        
        # Center region boundaries (in padded/cropped image coordinates)
        center_left = crop_left + context_margin_w
        center_right = center_left + center_w
        center_top = crop_top + context_margin_h
        center_bottom = center_top + center_h
        
        # Filter: keep only boxes whose center is inside CENTER REGION
        # Use half-open interval [left, right) to avoid tile boundary overlap
        mask = (
            (cx >= center_left) & (cx < center_right) &
            (cy >= center_top) & (cy < center_bottom)
        )
        
        if not mask.any():
            # No boxes left, return empty target
            empty_target = {}
            for key, value in target.items():
                if key == "boxes":
                    if isinstance(boxes, BoundingBoxes):
                        # Handle different torchvision versions
                        boxes_kwargs = {"format": boxes.format}
                        if _boxes_keys[1] == "canvas_size":
                            boxes_kwargs["canvas_size"] = (crop_bottom - crop_top, crop_right - crop_left)
                        else:
                            boxes_kwargs["spatial_size"] = (crop_bottom - crop_top, crop_right - crop_left)
                        
                        empty_target[key] = BoundingBoxes(
                            torch.zeros((0, 4), dtype=boxes.dtype, device=boxes.device),
                            **boxes_kwargs
                        )
                    else:
                        empty_target[key] = torch.zeros((0, 4), dtype=boxes.dtype, device=boxes.device)
                elif isinstance(value, torch.Tensor) and len(value) == len(boxes_xyxy):
                    # Preserve dtype and device for all tensor fields
                    empty_target[key] = torch.zeros((0,) + value.shape[1:], dtype=value.dtype, device=value.device)
                else:
                    empty_target[key] = value
            return empty_target
        
        # Clip boxes to crop region
        clipped_boxes = boxes_xyxy[mask].clone()
        clipped_boxes[:, 0] = torch.clamp(clipped_boxes[:, 0], crop_left, crop_right)
        clipped_boxes[:, 1] = torch.clamp(clipped_boxes[:, 1], crop_top, crop_bottom)
        clipped_boxes[:, 2] = torch.clamp(clipped_boxes[:, 2], crop_left, crop_right)
        clipped_boxes[:, 3] = torch.clamp(clipped_boxes[:, 3], crop_top, crop_bottom)
        
        # Shift boxes to new coordinate system (preserve dtype and device)
        offset = torch.tensor(
            [crop_left, crop_top, crop_left, crop_top], 
            dtype=clipped_boxes.dtype, 
            device=clipped_boxes.device
        )
        clipped_boxes = clipped_boxes - offset
        
        # Convert back to original format if needed
        crop_h = crop_bottom - crop_top
        crop_w = crop_right - crop_left
        
        if isinstance(boxes, BoundingBoxes):
            # Create BoundingBoxes with proper format (optimized: in-place conversion)
            if box_fmt == "xywh":
                # Convert back to xywh (xyxy -> xywh)
                w = clipped_boxes[:, 2] - clipped_boxes[:, 0]
                h = clipped_boxes[:, 3] - clipped_boxes[:, 1]
                clipped_boxes = clipped_boxes.clone()  # Avoid modifying original
                clipped_boxes[:, 2] = w
                clipped_boxes[:, 3] = h
            elif box_fmt == "cxcywh":
                # Convert back to cxcywh (xyxy -> cxcywh)
                w = clipped_boxes[:, 2] - clipped_boxes[:, 0]
                h = clipped_boxes[:, 3] - clipped_boxes[:, 1]
                cx = clipped_boxes[:, 0] + w / 2
                cy = clipped_boxes[:, 1] + h / 2
                clipped_boxes = clipped_boxes.clone()  # Avoid modifying original
                clipped_boxes[:, 0] = cx
                clipped_boxes[:, 1] = cy
                clipped_boxes[:, 2] = w
                clipped_boxes[:, 3] = h
            
            # Handle different torchvision versions
            boxes_kwargs = {"format": boxes.format}
            if _boxes_keys[1] == "canvas_size":
                boxes_kwargs["canvas_size"] = (crop_h, crop_w)
            else:
                boxes_kwargs["spatial_size"] = (crop_h, crop_w)
            
            clipped_boxes = BoundingBoxes(clipped_boxes, **boxes_kwargs)
        
        # Update target with filtered and clipped boxes
        filtered_target = {}
        for key, value in target.items():
            if key == "boxes":
                filtered_target[key] = clipped_boxes
            elif isinstance(value, torch.Tensor) and len(value) == len(boxes_xyxy):
                filtered_target[key] = value[mask]
            else:
                filtered_target[key] = value
        
        return filtered_target
    
    def _add_dummy_box(self, image, target):
        """
        Add a dummy 3x3 bounding box at a random location within CENTER GT REGION.
        Called when no bounding boxes remain after cropping.
        
        The dummy box is placed randomly within the center GT region to ensure
        it's in the valid annotation area.
        """
        # Get final image size (optimized)
        if isinstance(image, PIL.Image.Image):
            img_w, img_h = image.size
        elif isinstance(image, torch.Tensor):
            img_h, img_w = image.shape[-2:]  # Works for both 3D and 4D tensors
        elif hasattr(image, 'spatial_size'):
            img_h, img_w = image.spatial_size
        else:
            # Default to crop_size
            img_h, img_w = self.crop_size, self.crop_size
        
        # Calculate center GT region boundaries
        # Center GT region: [context_margin, context_margin] to [context_margin + center_gt_size, context_margin + center_gt_size]
        context_margin = (self.crop_size - self.center_gt_size) // 2
        center_gt_x1 = context_margin
        center_gt_y1 = context_margin
        center_gt_x2 = context_margin + self.center_gt_size
        center_gt_y2 = context_margin + self.center_gt_size
        
        # Random location for 3x3 box within center GT region
        max_x = center_gt_x2 - 3
        max_y = center_gt_y2 - 3
        
        if max_x < center_gt_x1 or max_y < center_gt_y1:
            # Center GT region too small for 3x3 box, place at center GT start
            x1, y1 = center_gt_x1, center_gt_y1
            x2, y2 = min(center_gt_x1 + 3, center_gt_x2), min(center_gt_y1 + 3, center_gt_y2)
        else:
            x1 = int(torch.randint(center_gt_x1, max_x + 1, (1,)).item())
            y1 = int(torch.randint(center_gt_y1, max_y + 1, (1,)).item())
            x2 = x1 + 3
            y2 = y1 + 3
        
        # Create dummy box in xyxy format
        dummy_box = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
        
        # Get original boxes format if available
        boxes = target.get("boxes")
        if boxes is not None and isinstance(boxes, BoundingBoxes):
            # Create BoundingBoxes with same format
            boxes_kwargs = {"format": boxes.format}
            if _boxes_keys[1] == "canvas_size":
                boxes_kwargs["canvas_size"] = (img_h, img_w)
            else:
                boxes_kwargs["spatial_size"] = (img_h, img_w)
            
            # Convert to original format if needed
            box_fmt = boxes.format.value.lower()
            if box_fmt == "xywh":
                # Convert xyxy to xywh
                dummy_box[0, 2] = dummy_box[0, 2] - dummy_box[0, 0]  # w
                dummy_box[0, 3] = dummy_box[0, 3] - dummy_box[0, 1]  # h
            elif box_fmt == "cxcywh":
                # Convert xyxy to cxcywh
                w = dummy_box[0, 2] - dummy_box[0, 0]
                h = dummy_box[0, 3] - dummy_box[0, 1]
                cx = dummy_box[0, 0] + w / 2
                cy = dummy_box[0, 1] + h / 2
                dummy_box = torch.tensor([[cx, cy, w, h]], dtype=torch.float32)
            
            dummy_box = BoundingBoxes(dummy_box, **boxes_kwargs)
        
        # Update target with dummy box
        new_target = {}
        for key, value in target.items():
            if key == "boxes":
                new_target[key] = dummy_box
            elif key == "labels":
                # Add dummy label (0 or background class)
                if isinstance(value, torch.Tensor):
                    new_target[key] = torch.tensor([0], dtype=value.dtype)
                else:
                    new_target[key] = torch.tensor([0], dtype=torch.int64)
            elif isinstance(value, torch.Tensor) and len(value) > 0:
                # Add dummy entry for other fields (e.g., area, iscrowd)
                if value.dim() == 1:
                    new_target[key] = torch.zeros(1, dtype=value.dtype)
                else:
                    new_target[key] = torch.zeros((1,) + value.shape[1:], dtype=value.dtype)
            else:
                new_target[key] = value
        
        return new_target
    
    def _reflect_boxes_horizontal(self, boxes_xyxy, orig_width, pad_left, pad_right):
        """
        Reflect boxes horizontally for Stage 1 padding (2-stage reflection).
        
        Args:
            boxes_xyxy: Original boxes in XYXY format [N, 4]
            orig_width: Original image width before padding
            pad_left: Left padding amount
            pad_right: Right padding amount
        
        Returns:
            Reflected boxes in XYXY format [M, 4] where M can be 0
        """
        reflected_boxes = []
        
        # Left reflection: boxes near left edge get reflected
        if pad_left > 0:
            # Find boxes within reflection range [0, pad_left]
            mask = boxes_xyxy[:, 0] < pad_left  # x1 < pad_left
            if mask.any():
                left_boxes = boxes_xyxy[mask].clone()
                # Reflect across x=0: x' = -x
                reflected_x1 = -left_boxes[:, 2]  # Swap and negate
                reflected_x2 = -left_boxes[:, 0]
                left_boxes[:, 0] = reflected_x1
                left_boxes[:, 2] = reflected_x2
                reflected_boxes.append(left_boxes)
        
        # Right reflection: boxes near right edge get reflected
        if pad_right > 0:
            # Find boxes within reflection range [orig_width - pad_right, orig_width]
            mask = boxes_xyxy[:, 2] > (orig_width - pad_right)  # x2 > orig_width - pad_right
            if mask.any():
                right_boxes = boxes_xyxy[mask].clone()
                # Reflect across x=orig_width: x' = 2*orig_width - x
                reflected_x1 = 2 * orig_width - right_boxes[:, 2]  # Swap and reflect
                reflected_x2 = 2 * orig_width - right_boxes[:, 0]
                right_boxes[:, 0] = reflected_x1
                right_boxes[:, 2] = reflected_x2
                reflected_boxes.append(right_boxes)
        
        if len(reflected_boxes) > 0:
            return torch.cat(reflected_boxes, dim=0)
        else:
            return torch.zeros((0, 4), dtype=boxes_xyxy.dtype, device=boxes_xyxy.device)
    
    def _reflect_boxes_vertical(self, boxes_xyxy, orig_height, pad_top, pad_bottom):
        """
        Reflect boxes vertically for Stage 1 padding (2-stage reflection).
        
        Args:
            boxes_xyxy: Original boxes in XYXY format [N, 4]
            orig_height: Original image height before padding
            pad_top: Top padding amount
            pad_bottom: Bottom padding amount
        
        Returns:
            Reflected boxes in XYXY format [M, 4] where M can be 0
        """
        reflected_boxes = []
        
        # Top reflection: boxes near top edge get reflected
        if pad_top > 0:
            # Find boxes within reflection range [0, pad_top]
            mask = boxes_xyxy[:, 1] < pad_top  # y1 < pad_top
            if mask.any():
                top_boxes = boxes_xyxy[mask].clone()
                # Reflect across y=0: y' = -y
                reflected_y1 = -top_boxes[:, 3]  # Swap and negate
                reflected_y2 = -top_boxes[:, 1]
                top_boxes[:, 1] = reflected_y1
                top_boxes[:, 3] = reflected_y2
                reflected_boxes.append(top_boxes)
        
        # Bottom reflection: boxes near bottom edge get reflected
        if pad_bottom > 0:
            # Find boxes within reflection range [orig_height - pad_bottom, orig_height]
            mask = boxes_xyxy[:, 3] > (orig_height - pad_bottom)  # y2 > orig_height - pad_bottom
            if mask.any():
                bottom_boxes = boxes_xyxy[mask].clone()
                # Reflect across y=orig_height: y' = 2*orig_height - y
                reflected_y1 = 2 * orig_height - bottom_boxes[:, 3]  # Swap and reflect
                reflected_y2 = 2 * orig_height - bottom_boxes[:, 1]
                bottom_boxes[:, 1] = reflected_y1
                bottom_boxes[:, 3] = reflected_y2
                reflected_boxes.append(bottom_boxes)
        
        if len(reflected_boxes) > 0:
            return torch.cat(reflected_boxes, dim=0)
        else:
            return torch.zeros((0, 4), dtype=boxes_xyxy.dtype, device=boxes_xyxy.device)
    
    def _reflect_boxes_left(self, boxes_xyxy, img_width, pad_left):
        """
        Reflect boxes across left boundary for Stage 2 padding.
        
        Args:
            boxes_xyxy: Boxes in XYXY format [N, 4]
            img_width: Current image width
            pad_left: Left padding amount
        
        Returns:
            tuple: (reflected boxes in XYXY format [M, 4], mask indicating which boxes were reflected)
        """
        if pad_left == 0:
            return torch.zeros((0, 4), dtype=boxes_xyxy.dtype, device=boxes_xyxy.device), torch.zeros(len(boxes_xyxy), dtype=torch.bool, device=boxes_xyxy.device)
        
        # Boxes within [0, pad_left] range get reflected to [-pad_left, 0]
        mask = boxes_xyxy[:, 0] < pad_left
        if not mask.any():
            return torch.zeros((0, 4), dtype=boxes_xyxy.dtype, device=boxes_xyxy.device), mask
        
        reflected = boxes_xyxy[mask].clone()
        # Reflect across x=0: x' = -x
        reflected_x1 = -reflected[:, 2]
        reflected_x2 = -reflected[:, 0]
        reflected[:, 0] = reflected_x1
        reflected[:, 2] = reflected_x2
        
        return reflected, mask
    
    def _reflect_boxes_right(self, boxes_xyxy, img_width, pad_right):
        """
        Reflect boxes across right boundary for Stage 2 padding.
        
        Args:
            boxes_xyxy: Boxes in XYXY format [N, 4]
            img_width: Current image width
            pad_right: Right padding amount
        
        Returns:
            tuple: (reflected boxes in XYXY format [M, 4], mask indicating which boxes were reflected)
        """
        if pad_right == 0:
            return torch.zeros((0, 4), dtype=boxes_xyxy.dtype, device=boxes_xyxy.device), torch.zeros(len(boxes_xyxy), dtype=torch.bool, device=boxes_xyxy.device)
        
        # Boxes within [img_width - pad_right, img_width] range get reflected
        mask = boxes_xyxy[:, 2] > (img_width - pad_right)
        if not mask.any():
            return torch.zeros((0, 4), dtype=boxes_xyxy.dtype, device=boxes_xyxy.device), mask
        
        reflected = boxes_xyxy[mask].clone()
        # Reflect across x=img_width: x' = 2*img_width - x
        reflected_x1 = 2 * img_width - reflected[:, 2]
        reflected_x2 = 2 * img_width - reflected[:, 0]
        reflected[:, 0] = reflected_x1
        reflected[:, 2] = reflected_x2
        
        return reflected, mask
    
    def _reflect_boxes_top(self, boxes_xyxy, img_height, pad_top):
        """
        Reflect boxes across top boundary for Stage 2 padding.
        
        Args:
            boxes_xyxy: Boxes in XYXY format [N, 4]
            img_height: Current image height
            pad_top: Top padding amount
        
        Returns:
            tuple: (reflected boxes in XYXY format [M, 4], mask indicating which boxes were reflected)
        """
        if pad_top == 0:
            return torch.zeros((0, 4), dtype=boxes_xyxy.dtype, device=boxes_xyxy.device), torch.zeros(len(boxes_xyxy), dtype=torch.bool, device=boxes_xyxy.device)
        
        # Boxes within [0, pad_top] range get reflected to [-pad_top, 0]
        mask = boxes_xyxy[:, 1] < pad_top
        if not mask.any():
            return torch.zeros((0, 4), dtype=boxes_xyxy.dtype, device=boxes_xyxy.device), mask
        
        reflected = boxes_xyxy[mask].clone()
        # Reflect across y=0: y' = -y
        reflected_y1 = -reflected[:, 3]
        reflected_y2 = -reflected[:, 1]
        reflected[:, 1] = reflected_y1
        reflected[:, 3] = reflected_y2
        
        return reflected, mask
    
    def _reflect_boxes_bottom(self, boxes_xyxy, img_height, pad_bottom):
        """
        Reflect boxes across bottom boundary for Stage 2 padding.
        
        Args:
            boxes_xyxy: Boxes in XYXY format [N, 4]
            img_height: Current image height
            pad_bottom: Bottom padding amount
        
        Returns:
            tuple: (reflected boxes in XYXY format [M, 4], mask indicating which boxes were reflected)
        """
        if pad_bottom == 0:
            return torch.zeros((0, 4), dtype=boxes_xyxy.dtype, device=boxes_xyxy.device), torch.zeros(len(boxes_xyxy), dtype=torch.bool, device=boxes_xyxy.device)
        
        # Boxes within [img_height - pad_bottom, img_height] range get reflected
        mask = boxes_xyxy[:, 3] > (img_height - pad_bottom)
        if not mask.any():
            return torch.zeros((0, 4), dtype=boxes_xyxy.dtype, device=boxes_xyxy.device), mask
        
        reflected = boxes_xyxy[mask].clone()
        # Reflect across y=img_height: y' = 2*img_height - y
        reflected_y1 = 2 * img_height - reflected[:, 3]
        reflected_y2 = 2 * img_height - reflected[:, 1]
        reflected[:, 1] = reflected_y1
        reflected[:, 3] = reflected_y2
        
        return reflected, mask
