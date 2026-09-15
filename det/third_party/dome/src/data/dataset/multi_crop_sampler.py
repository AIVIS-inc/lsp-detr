"""
Custom Distributed Sampler for MultiCropDatasetWrapper

This sampler ensures balanced distribution of crops across GPUs
by shuffling crop indices instead of sequential distribution.
"""

import math
import torch
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


class MultiCropDistributedSampler(DistributedSampler):
    """
    Distributed sampler optimized for MultiCropDatasetWrapper.
    
    Ensures each GPU gets a balanced mix of crops from different images
    rather than sequential blocks, which can cause memory imbalance.
    
    Args:
        dataset: MultiCropDatasetWrapper instance
        num_replicas: Number of processes (GPUs)
        rank: Rank of current process
        shuffle: Whether to shuffle indices
        seed: Random seed for reproducibility
        drop_last: Drop last incomplete batch
    """
    
    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        drop_last=False,
    ):
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        
        # Check if dataset is MultiCropDatasetWrapper
        self.is_multicrop = hasattr(dataset, 'crops_per_image')
        self.is_adaptive = False  # Default value
        
        if self.is_multicrop:
            self.crops_per_image = dataset.crops_per_image
            self.num_original_images = len(dataset.dataset)
            
            # Check if dataset uses adaptive cropping (different crop counts per image)
            self.is_adaptive = hasattr(dataset, 'image_crop_counts')
            
            if self.is_adaptive:
                self.image_crop_counts = dataset.image_crop_counts
                self.image_crop_offsets = dataset.image_crop_offsets
                print(f"[Rank {self.rank}] MultiCropDistributedSampler initialized (adaptive mode):")
                print(f"  - Original images: {self.num_original_images}")
                print(f"  - Total crops: {len(dataset)}")
                print(f"  - Samples per GPU: {self.num_samples}")
            else:
                print(f"[Rank {self.rank}] MultiCropDistributedSampler initialized:")
                print(f"  - Original images: {self.num_original_images}")
                print(f"  - Crops per image: {self.crops_per_image}")
                print(f"  - Total samples: {len(dataset)}")
                print(f"  - Samples per GPU: {self.num_samples}")
    
    def __iter__(self):
        if self.shuffle:
            # Create shuffled indices with better distribution
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            if self.is_multicrop and self.is_adaptive:
                # Adaptive mode: Different images have different crop counts
                # Strategy: Shuffle original images, then enumerate their crops
                
                # Shuffle original image indices
                orig_indices = torch.randperm(self.num_original_images, generator=g).tolist()
                
                # Create crop indices based on actual crop counts
                indices = []
                for orig_idx in orig_indices:
                    offset = self.image_crop_offsets[orig_idx]
                    num_crops = self.image_crop_counts[orig_idx]
                    # Add all crops for this image
                    for crop_idx in range(num_crops):
                        global_idx = offset + crop_idx
                        indices.append(global_idx)
                
                # Final shuffle for better batch diversity
                indices_tensor = torch.tensor(indices)
                shuffle_idx = torch.randperm(len(indices), generator=g)
                indices = indices_tensor[shuffle_idx].tolist()
                
            elif self.is_multicrop:
                # Fixed crop count mode: All images have same crop count
                # Strategy: Interleave crops from different images
                # This maximizes batch diversity
                
                # Shuffle original image indices
                orig_indices = torch.randperm(self.num_original_images, generator=g).tolist()
                
                # Create interleaved crop indices for better batch diversity
                indices = []
                for crop_idx in range(self.crops_per_image):
                    # For each crop position, iterate through all images
                    for orig_idx in orig_indices:
                        global_idx = orig_idx * self.crops_per_image + crop_idx
                        indices.append(global_idx)
                
                # Final shuffle to break any remaining patterns
                indices_tensor = torch.tensor(indices)
                shuffle_idx = torch.randperm(len(indices), generator=g)
                indices = indices_tensor[shuffle_idx].tolist()
            else:
                # Standard shuffling for non-multicrop datasets
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible
            indices = indices[:self.total_size]
        
        assert len(indices) == self.total_size
        
        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        
        return iter(indices)

