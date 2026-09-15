"""
Multi-Crop Dataset Wrapper for Large Image Training

This wrapper expands each sample in the original dataset into N crops,
enabling better annotation utilization for large images (e.g., 4096x4096).

Copyright (c) 2024. All Rights Reserved.
"""

from typing import Optional
import math
import torch
from torch.utils.data import Dataset
from ...core import register

__all__ = ["MultiCropDatasetWrapper"]


@register()
class MultiCropDatasetWrapper(Dataset):
    """    
    Wrapper that expands each dataset sample into multiple crops with adaptive crop count.
    
    Adaptive Crop Strategy:
    - Small images (< crop_size): 1 crop (padded + center crop)
    - Medium images (≥ crop_size but < crop_size * large_threshold_multiplier): 3 crops
    - Large images (≥ crop_size * large_threshold_multiplier): crops_per_image crops
    
    For 4096x4096 images with crop_size=2048, large_threshold_multiplier=2, and center_gt_size=672:
    - stride = center_gt_size = 672
    - Total grid points: ceil(4096/672) x ceil(4096/672) = 7x7 = 49
    - Image size threshold: 2048 * 2 = 4096
    - Since 4096 ≥ 4096, this image gets crops_per_image crops (large image)
    
    Args:
        dataset: Original dataset (e.g., CocoDetection)
        crops_per_image: Max number of crops for large images (default: 4)
        grid_selection_strategy: How to select grid indices
            - 'random': Randomly select N grids per image (default)
            - 'stratified': Evenly distributed across image
            - 'deterministic': Use fixed grid pattern per image
        seed: Random seed for reproducibility (optional)
        crop_size: Crop size (auto-detected from transforms if not provided)
        large_threshold_multiplier: Multiplier for large image threshold (default: 2)
    
    Example:
        >>> # Original dataset has 1000 images (mix of sizes)
        >>> base_dataset = CocoDetection(...)
        >>> wrapped_dataset = MultiCropDatasetWrapper(base_dataset, crops_per_image=18, crop_size=2048, large_threshold_multiplier=2)
        >>> # Small images (< 2048): 1 crop each
        >>> # Medium images (2048 ≤ size < 4096): 3 crops each
        >>> # Large images (≥ 4096): 18 crops each
    """
    
    __inject__ = ['dataset']
    __share__ = []
    
    def __init__(
        self,
        dataset: Dataset = None,
        crops_per_image: int = 4,
        grid_selection_strategy: str = 'random',
        seed: Optional[int] = None,
        crop_size: Optional[int] = None,
        large_threshold_multiplier: int = 2,
        **kwargs
    ):
        super().__init__()
        
        # If dataset is not provided, create it from kwargs (for config compatibility)
        if dataset is None and 'type' in kwargs:
            from ...core import create
            dataset_config = kwargs.copy()
            dataset = create('dataset', {'dataset': dataset_config})
        
        if dataset is None:
            raise ValueError("Either 'dataset' or dataset creation kwargs must be provided")
        
        self.dataset = dataset
        self.crops_per_image = crops_per_image
        self.grid_selection_strategy = grid_selection_strategy
        self.seed = seed
        self.large_threshold_multiplier = large_threshold_multiplier
        
        # Auto-detect crop_size from transforms if not explicitly provided
        if crop_size is None:
            self.crop_size = self._detect_crop_size_from_transforms()
            if self.crop_size is None:
                self.crop_size = 2048  # Default fallback
                print(f"[MultiCropDatasetWrapper] Warning: Could not detect crop_size from transforms, using default {self.crop_size}")
            else:
                print(f"[MultiCropDatasetWrapper] Auto-detected crop_size: {self.crop_size}")
        else:
            self.crop_size = crop_size
        
        # For deterministic grid selection
        if seed is not None:
            self.rng = torch.Generator()
            self.rng.manual_seed(seed)
        else:
            self.rng = None
        
        # Precompute image-specific crops (adaptive based on image size)
        self._compute_image_crops()
        
        # Precompute grid indices for each original sample (for stratified/deterministic)
        if grid_selection_strategy in ['stratified', 'deterministic']:
            self._precompute_grid_indices()
    
    def __len__(self):
        """Return total number of crops (sum of per-image crops)"""
        return sum(self.image_crop_counts.values())
    
    def __getitem__(self, idx):
        """
        Get crop from the dataset (adaptive based on image size).
        
        Args:
            idx: Global crop index (0 to len(self)-1)
        
        Returns:
            (image, target) tuple where target contains 'grid_index' for cropping
        """
        try:
            # Find which original image this crop belongs to
            original_idx, crop_idx = self._global_to_local_idx(idx)
            
            # Get original sample (PIL Image, dict)
            image, target = self.dataset.load_item(original_idx)
            
            # Verify types
            if not hasattr(image, 'size') and not hasattr(image, 'shape'):
                raise TypeError(f"Expected PIL Image or Tensor, got {type(image)}")
            if not isinstance(target, dict):
                raise TypeError(f"Expected dict for target, got {type(target)}")
            
            # Check if this is a small image (no cropping needed)
            img_w, img_h = image.size if hasattr(image, 'size') else (image.shape[-1], image.shape[-2])
            is_small_image = img_w <= self.crop_size or img_h <= self.crop_size
            
            if is_small_image:
                # Small image - no grid_index needed (will be padded and center-cropped)
                grid_index = None
            else:
                # Large image - determine grid index for this crop
                grid_index = self._get_grid_index(original_idx, crop_idx, image)
            
            # Add metadata to target
            target['grid_index'] = grid_index
            target['crop_idx'] = crop_idx
            target['original_idx'] = original_idx
            target['is_small_image'] = is_small_image
            
            # Apply transforms (including RandomCropWithGrid which will use grid_index)
            # Dome-DETR transforms return (image, target, dataset) tuple
            if hasattr(self.dataset, '_transforms') and self.dataset._transforms is not None:
                result = self.dataset._transforms(image, target, self.dataset)
                # Handle the 3-tuple return from Dome-DETR transforms
                if isinstance(result, tuple) and len(result) == 3:
                    image, target, _ = result
                elif isinstance(result, tuple) and len(result) == 2:
                    image, target = result
                else:
                    raise TypeError(f"Unexpected transform output: {type(result)}, len={len(result) if hasattr(result, '__len__') else 'N/A'}")
            
            # Final type check
            if not isinstance(target, dict):
                raise TypeError(f"After transforms, target should be dict, got {type(target)}")
            
            # Return as tuple
            return (image, target)
            
        except Exception as e:
            print(f"[MultiCropDatasetWrapper] Error in __getitem__ for idx={idx}:")
            print(f"  - original_idx: {original_idx if 'original_idx' in locals() else 'N/A'}")
            print(f"  - crop_idx: {crop_idx if 'crop_idx' in locals() else 'N/A'}")
            print(f"  - image type: {type(image) if 'image' in locals() else 'N/A'}")
            print(f"  - target type: {type(target) if 'target' in locals() else 'N/A'}")
            print(f"  - Error: {e}")
            raise
    
    def _detect_crop_size_from_transforms(self):
        """
        Auto-detect crop_size from transform pipeline.
        Looks for RandomCropWithGrid transform and extracts crop_size.
        """
        if not hasattr(self.dataset, '_transforms'):
            return None
        
        transforms = self.dataset._transforms
        if transforms is None:
            return None
        
        # Check if it's a Compose transform
        if hasattr(transforms, 'transforms'):
            for transform in transforms.transforms:
                # Look for RandomCropWithGrid
                if transform.__class__.__name__ == 'RandomCropWithGrid':
                    if hasattr(transform, 'crop_size'):
                        return transform.crop_size
        
        # Single transform case
        if hasattr(transforms, 'crop_size'):
            return transforms.crop_size
        
        return None
    
    def _compute_image_crops(self):
        """
        Precompute number of crops for each image based on size.
        - Small images (< crop_size): 1 crop (padded + center crop)
        - Medium images (≥ crop_size but < crop_size * large_threshold_multiplier): 3 crops
        - Large images (≥ crop_size * large_threshold_multiplier): crops_per_image crops
        
        Optimized version: Uses COCO metadata instead of loading actual images.
        """
        self.image_crop_counts = {}  # {original_idx: num_crops}
        self.image_crop_offsets = {}  # {original_idx: starting_global_idx}
        
        total_crops = 0
        
        # Size thresholds (configurable via large_threshold_multiplier)
        large_threshold = self.crop_size * self.large_threshold_multiplier
        
        # Check if dataset has COCO metadata (fast path)
        use_metadata = hasattr(self.dataset, 'coco') and hasattr(self.dataset, 'ids')
        
        if use_metadata:
            print(f"[MultiCropDatasetWrapper] Using COCO metadata for fast initialization...")
            for idx in range(len(self.dataset)):
                # Get image metadata without loading the actual image
                image_id = self.dataset.ids[idx]
                img_info = self.dataset.coco.loadImgs(image_id)[0]
                img_w = img_info['width']
                img_h = img_info['height']
                
                # Determine number of crops based on image size
                if img_w <= self.crop_size or img_h <= self.crop_size:
                    # Small images: 1 crop (no multi-cropping needed)
                    num_crops = 1
                elif img_w < large_threshold or img_h < large_threshold:
                    # Medium images: limited crops (3 crops)
                    num_crops = 3
                else:
                    # Large images: full multi-cropping
                    num_crops = self.crops_per_image
                
                self.image_crop_counts[idx] = num_crops
                self.image_crop_offsets[idx] = total_crops
                total_crops += num_crops
        else:
            # Fallback: Load images to check size (slower)
            print(f"[MultiCropDatasetWrapper] Warning: COCO metadata not available, loading images to check sizes...")
            for idx in range(len(self.dataset)):
                # Get image to check size
                if hasattr(self.dataset, 'load_item'):
                    image, _ = self.dataset.load_item(idx)
                else:
                    image, _ = self.dataset[idx]
                
                img_w, img_h = image.size if hasattr(image, 'size') else (image.shape[-1], image.shape[-2])
                
                # Determine number of crops based on image size
                if img_w <= self.crop_size or img_h <= self.crop_size:
                    num_crops = 1
                elif img_w < large_threshold or img_h < large_threshold:
                    num_crops = 3
                else:
                    num_crops = self.crops_per_image
                
                self.image_crop_counts[idx] = num_crops
                self.image_crop_offsets[idx] = total_crops
                total_crops += num_crops
        
        print(f"[MultiCropDatasetWrapper] Image crop analysis:")
        print(f"  - Size thresholds: small < {self.crop_size}, medium < {large_threshold}, large ≥ {large_threshold}")
        small_images = sum(1 for c in self.image_crop_counts.values() if c == 1)
        medium_images = sum(1 for c in self.image_crop_counts.values() if c == 3)
        large_images = sum(1 for c in self.image_crop_counts.values() if c > 3)
        print(f"  - Small images (< {self.crop_size}): {small_images} (1 crop each)")
        print(f"  - Medium images ({self.crop_size} ≤ size < {large_threshold}): {medium_images} (3 crops each)")
        print(f"  - Large images (≥ {large_threshold}): {large_images} ({self.crops_per_image} crops each)")
        print(f"  - Total crops: {total_crops}")
    
    def _global_to_local_idx(self, global_idx: int):
        """
        Convert global crop index to (original_image_idx, crop_idx).
        """
        for original_idx in range(len(self.dataset)):
            offset = self.image_crop_offsets[original_idx]
            num_crops = self.image_crop_counts[original_idx]
            
            if global_idx < offset + num_crops:
                crop_idx = global_idx - offset
                return original_idx, crop_idx
        
        raise IndexError(f"Global index {global_idx} out of range")
    
    def _get_grid_index(self, original_idx: int, crop_idx: int, image) -> Optional[int]:
        """
        Determine which grid index to use for this crop.
        """
        if self.grid_selection_strategy == 'random':
            return None
        
        elif self.grid_selection_strategy == 'stratified':
            return self.grid_indices[original_idx][crop_idx]
        
        elif self.grid_selection_strategy == 'deterministic':
            return self.grid_indices[original_idx][crop_idx]
        
        else:
            raise ValueError(f"Unknown grid_selection_strategy: {self.grid_selection_strategy}")
    
    def _precompute_grid_indices(self):
        """
        Precompute grid indices for stratified or deterministic selection.
        """
        self.grid_indices = {}
        
        # Check if dataset has COCO metadata (fast path)
        use_metadata = hasattr(self.dataset, 'coco') and hasattr(self.dataset, 'ids')
        
        for idx in range(len(self.dataset)):
            # Get image-specific crop count
            num_crops_for_image = self.image_crop_counts.get(idx, self.crops_per_image)
            
            # Get image size
            if use_metadata:
                image_id = self.dataset.ids[idx]
                img_info = self.dataset.coco.loadImgs(image_id)[0]
                img_w = img_info['width']
                img_h = img_info['height']
            else:
                if hasattr(self.dataset, 'load_item'):
                    image, _ = self.dataset.load_item(idx)
                else:
                    image, _ = self.dataset[idx]
                img_w, img_h = image.size if hasattr(image, 'size') else (4096, 4096)
            
            # Get center_gt_size from RandomCropWithGrid transform
            center_gt_size = None
            if hasattr(self.dataset, '_transforms') and self.dataset._transforms is not None:
                transforms = self.dataset._transforms
                if hasattr(transforms, 'transforms'):
                    for transform in transforms.transforms:
                        if transform.__class__.__name__ == 'RandomCropWithGrid':
                            if hasattr(transform, 'center_gt_size'):
                                center_gt_size = transform.center_gt_size
                                break
            
            if center_gt_size is None:
                center_gt_size = self.crop_size // 4
            
            stride = center_gt_size
            
            # Calculate valid range
            max_valid_ix = int((img_w - center_gt_size) // stride)
            max_valid_iy = int((img_h - center_gt_size) // stride)
            
            nx = max(1, max_valid_ix + 1)
            ny = max(1, max_valid_iy + 1)
            
            total_grids = nx * ny
            
            if self.grid_selection_strategy == 'stratified':
                if total_grids <= num_crops_for_image:
                    selected = list(range(total_grids))
                    while len(selected) < num_crops_for_image:
                        selected.append(selected[-1])
                else:
                    step = total_grids / num_crops_for_image
                    selected = [int(i * step) for i in range(num_crops_for_image)]
            
            else:  # deterministic
                if self.rng is not None:
                    perm = torch.randperm(total_grids, generator=self.rng)
                else:
                    local_rng = torch.Generator()
                    local_rng.manual_seed(idx)
                    perm = torch.randperm(total_grids, generator=local_rng)
                
                selected = perm[:num_crops_for_image].tolist()
            
            self.grid_indices[idx] = selected
    
    def set_epoch(self, epoch: int):
        """Set epoch for epoch-dependent behavior"""
        if hasattr(self.dataset, 'set_epoch'):
            self.dataset.set_epoch(epoch)
        
        if self.rng is not None and self.grid_selection_strategy == 'random':
            self.rng.manual_seed(self.seed + epoch)
    
    @property
    def categories(self):
        if hasattr(self.dataset, 'categories'):
            return self.dataset.categories
        return None
    
    @property
    def category2name(self):
        if hasattr(self.dataset, 'category2name'):
            return self.dataset.category2name
        return None
    
    @property
    def category2label(self):
        if hasattr(self.dataset, 'category2label'):
            return self.dataset.category2label
        return None
    
    @property
    def label2category(self):
        if hasattr(self.dataset, 'label2category'):
            return self.dataset.label2category
        return None
    
    def extra_repr(self) -> str:
        s = f" crops_per_image (max): {self.crops_per_image}\n"
        s += f" crop_size: {self.crop_size}\n"
        s += f" large_threshold_multiplier: {self.large_threshold_multiplier}\n"
        s += f" large_threshold: {self.crop_size * self.large_threshold_multiplier}\n"
        s += f" grid_selection_strategy: {self.grid_selection_strategy}\n"
        s += f" original_dataset_size: {len(self.dataset)}\n"
        s += f" expanded_dataset_size: {len(self)}\n"
        
        if hasattr(self, 'image_crop_counts'):
            small_images = sum(1 for c in self.image_crop_counts.values() if c == 1)
            medium_images = sum(1 for c in self.image_crop_counts.values() if c == 3)
            large_images = sum(1 for c in self.image_crop_counts.values() if c > 3)
            s += f" crop_distribution: {small_images} small (1 crop), {medium_images} medium (3 crops), {large_images} large ({self.crops_per_image} crops)\n"
        
        s += f" wrapped_dataset:\n   {repr(self.dataset)}"
        return s
