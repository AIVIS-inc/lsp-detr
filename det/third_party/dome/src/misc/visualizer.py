""" "
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import PIL
import numpy as np
import torch
import torch.utils.data
import torchvision
from typing import List, Dict

torchvision.disable_beta_transforms_warning()

__all__ = ["show_sample", "save_samples"]


def save_samples(samples: torch.Tensor, targets: List[Dict], output_dir: str, split: str, normalized: bool, box_fmt: str, rank_prefix: int = None):
    '''
    normalized: whether the boxes are normalized to [0, 1]
    box_fmt: 'xyxy', 'xywh', 'cxcywh', D-FINE uses 'cxcywh' for training, 'xyxy' for validation
    rank_prefix: GPU rank for filename prefix (optional, for multi-GPU training)
    '''
    from torchvision.transforms.functional import to_pil_image
    from torchvision.ops import box_convert
    from pathlib import Path
    from PIL import ImageDraw, ImageFont
    import os

    # Create directory structure: output_dir/train_samples/train_epoch000_samples or output_dir/val_samples/val_samples
    os.makedirs(Path(output_dir) / Path(f"{split}_samples"), exist_ok=True)
    # Predefined colors (standard color names recognized by PIL)
    BOX_COLORS = [
        "red", "blue", "green", "orange", "purple",
        "cyan", "magenta", "yellow", "lime", "pink",
        "teal", "lavender", "brown", "beige", "maroon",
        "navy", "olive", "coral", "turquoise", "gold"
    ]

    LABEL_TEXT_COLOR = "white"

    font = ImageFont.load_default()
    font.size = 32

    for i, (sample, target) in enumerate(zip(samples, targets)):
        sample_visualization = sample.clone().cpu()
        target_boxes = target["boxes"].clone().cpu()
        target_labels = target["labels"].clone().cpu()
        
        # Handle image_id robustly - it might be empty tensor, missing, or malformed
        if "image_id" in target and isinstance(target["image_id"], torch.Tensor):
            if target["image_id"].numel() > 0:  # Check if tensor is not empty
                target_image_id = target["image_id"].item()
            else:
                # Empty tensor - use fallback
                fallback = target.get("original_idx", i)
                target_image_id = fallback.item() if isinstance(fallback, torch.Tensor) and fallback.numel() > 0 else (fallback if isinstance(fallback, int) else i)
        else:
            # Missing or not a tensor - use fallback
            fallback = target.get("original_idx", i)
            target_image_id = fallback.item() if isinstance(fallback, torch.Tensor) and fallback.numel() > 0 else (fallback if isinstance(fallback, int) else i)
        
        # Handle image_path - may not exist in Dome-DETR targets
        if "image_path" in target:
            target_image_path = target["image_path"]
            target_image_path_stem = Path(target_image_path).stem
        else:
            # Fallback: use image_id as filename
            target_image_path_stem = f"image_{target_image_id}"

        sample_visualization = to_pil_image(sample_visualization)
        sample_visualization_w, sample_visualization_h = sample_visualization.size

        # normalized to pixel space
        if normalized:
            target_boxes[:, 0] = target_boxes[:, 0] * sample_visualization_w
            target_boxes[:, 2] = target_boxes[:, 2] * sample_visualization_w
            target_boxes[:, 1] = target_boxes[:, 1] * sample_visualization_h
            target_boxes[:, 3] = target_boxes[:, 3] * sample_visualization_h

        # any box format -> xyxy
        target_boxes = box_convert(target_boxes, in_fmt=box_fmt, out_fmt="xyxy")

        # clip to image size
        target_boxes[:, 0] = torch.clamp(target_boxes[:, 0], 0, sample_visualization_w)
        target_boxes[:, 1] = torch.clamp(target_boxes[:, 1], 0, sample_visualization_h)
        target_boxes[:, 2] = torch.clamp(target_boxes[:, 2], 0, sample_visualization_w)
        target_boxes[:, 3] = torch.clamp(target_boxes[:, 3], 0, sample_visualization_h)

        # Convert to numpy but keep float for precision
        target_boxes_float = target_boxes.numpy()
        target_labels = target_labels.numpy().astype(np.int32)

        draw = ImageDraw.Draw(sample_visualization)

        # draw target boxes
        for box, label in zip(target_boxes_float, target_labels):
            x1, y1, x2, y2 = box
            
            # Round to nearest integer instead of truncating
            x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
            
            # Skip boxes with zero width or height after rounding
            if x2 <= x1 or y2 <= y1:
                continue

            # Select color based on class ID
            box_color = BOX_COLORS[int(label) % len(BOX_COLORS)]

            # Draw box with thin line (width=1)
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=1)
            
            # No label text - class differentiated by color only

        # Add rank prefix to filename if provided (for multi-GPU training)
        if rank_prefix is not None:
            filename = f"rank{rank_prefix}_{target_image_id}_{target_image_path_stem}.webp"
        else:
            filename = f"{target_image_id}_{target_image_path_stem}.webp"
        
        save_path = Path(output_dir) / f"{split}_samples" / filename
        sample_visualization.save(save_path)


def show_sample(sample):
    """for coco dataset/dataloader"""
    import matplotlib.pyplot as plt
    from torchvision.transforms.v2 import functional as F
    from torchvision.utils import draw_bounding_boxes

    image, target = sample
    if isinstance(image, PIL.Image.Image):
        image = F.to_image_tensor(image)

    image = F.convert_dtype(image, torch.uint8)
    annotated_image = draw_bounding_boxes(image, target["boxes"], colors="yellow", width=3)

    fig, ax = plt.subplots()
    ax.imshow(annotated_image.permute(1, 2, 0).numpy())
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.tight_layout()
    fig.show()
    plt.show()
