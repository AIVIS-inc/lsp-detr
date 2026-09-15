"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ...core import register

__all__ = ["DomePostProcessor"]


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class DomePostProcessor(nn.Module):
    __share__ = ["num_classes", "use_focal_loss", "num_top_queries", "remap_mscoco_category", "use_nms", "nms_iou_threshold", "nms_score_threshold", "class_agnostic", "keep_topk_after_nms"]

    def __init__(
        self, 
        num_classes=80, 
        use_focal_loss=True, 
        num_top_queries=1500, 
        remap_mscoco_category=False, 
        score_thresh=0.1,
        use_nms=False,
        nms_iou_threshold=0.7,
        nms_score_threshold=0.01,
        class_agnostic=True,
        keep_topk_after_nms=None,
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False
        self.score_thresh = score_thresh
        self.use_nms = use_nms
        self.nms_iou_threshold = nms_iou_threshold
        self.nms_score_threshold = nms_score_threshold
        self.class_agnostic = class_agnostic
        self.keep_topk_after_nms = keep_topk_after_nms

    def extra_repr(self) -> str:
        return f"use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}, use_nms={self.use_nms}, class_agnostic={self.class_agnostic}"

    def _apply_nms(self, boxes, scores, labels):
        """
        Apply NMS to a single batch of detections.
        
        Args:
            boxes: [N, 4] tensor in xyxy format
            scores: [N] tensor
            labels: [N] tensor
        
        Returns:
            keep: [M] tensor of indices to keep after NMS (indices into input tensors)
        """
        if len(boxes) == 0:
            return torch.empty((0,), dtype=torch.long, device=boxes.device)
        
        # Apply score threshold before NMS
        score_threshold_mask = scores > self.nms_score_threshold
        if not score_threshold_mask.any():
            return torch.empty((0,), dtype=torch.long, device=boxes.device)
        
        boxes_filtered = boxes[score_threshold_mask]
        scores_filtered = scores[score_threshold_mask]
        labels_filtered = labels[score_threshold_mask]
        
        if len(boxes_filtered) == 0:
            return torch.empty((0,), dtype=torch.long, device=boxes.device)
        
        # Apply NMS
        if self.class_agnostic:
            # Class-agnostic NMS: ignore class labels, treat all detections as same class
            batch_ids = torch.zeros_like(labels_filtered)
            keep_after_nms = torchvision.ops.batched_nms(
                boxes_filtered, scores_filtered, batch_ids, self.nms_iou_threshold
            )
        else:
            # Class-aware NMS: apply NMS separately for each class
            keep_after_nms = torchvision.ops.batched_nms(
                boxes_filtered, scores_filtered, labels_filtered, self.nms_iou_threshold
            )
        
        # Apply keep_topk_after_nms if specified
        if self.keep_topk_after_nms is not None and len(keep_after_nms) > self.keep_topk_after_nms:
            keep_after_nms = keep_after_nms[:self.keep_topk_after_nms]
        
        # Map back to original indices (indices into score_threshold_mask)
        score_threshold_indices = torch.nonzero(score_threshold_mask, as_tuple=True)[0]
        return score_threshold_indices[keep_after_nms]

    def _process_batch(
        self, 
        batch_labels, 
        batch_boxes, 
        batch_scores,
        batch_all_class_scores=None,
        query_padding_mask=None,
        query_indices=None
    ):
        """
        Process a single batch: filter padded queries and apply NMS.
        
        Args:
            batch_labels: [K] tensor
            batch_boxes: [K, 4] tensor
            batch_scores: [K] tensor
            batch_all_class_scores: [K, num_classes] tensor or None
            query_padding_mask: [Q] tensor (True = padded) or None
            query_indices: [K] tensor of query indices, or None
        
        Returns:
            dict with filtered labels, boxes, scores, all_class_scores
        """
        num_score_classes = batch_all_class_scores.shape[-1] if batch_all_class_scores is not None else 0

        # CRITICAL FIX: Filter out predictions from padded queries
        # Case 1: Both padding mask and query indices are available (most common)
        if query_padding_mask is not None and query_indices is not None:
            # Check which indices point to valid queries
            valid_indices_mask = ~query_padding_mask[query_indices]  # True = valid
            # Filter by score threshold to remove near-zero scores from padded queries
            score_threshold = 1e-10  # Very small threshold to catch padded queries
            valid_score_mask = batch_scores > score_threshold
            keep_mask = valid_indices_mask & valid_score_mask
        # Case 2: Only padding mask available (when query_indices is None, e.g., all queries selected)
        elif query_padding_mask is not None:
            # When query_indices is None, it means all queries were selected (K == Q)
            # In this case, we can directly use the padding mask
            # batch_labels/boxes/scores shape should match query_padding_mask
            if len(batch_scores) == len(query_padding_mask):
                valid_mask = ~query_padding_mask  # True = valid
                score_threshold = 1e-10
                valid_score_mask = batch_scores > score_threshold
                keep_mask = valid_mask & valid_score_mask
            else:
                # Shape mismatch: fallback to score threshold only
                score_threshold = 1e-10
                keep_mask = batch_scores > score_threshold
        else:
            # No padding mask available: fallback to score threshold only
            score_threshold = 1e-10
            keep_mask = batch_scores > score_threshold
        
        if not keep_mask.any():
            return {
                "labels": torch.empty((0,), dtype=batch_labels.dtype, device=batch_labels.device),
                "boxes": torch.empty((0, 4), dtype=batch_boxes.dtype, device=batch_boxes.device),
                "scores": torch.empty((0,), dtype=batch_scores.dtype, device=batch_scores.device),
                "all_class_scores": torch.empty((0, num_score_classes), dtype=batch_scores.dtype, device=batch_scores.device),
            }
        
        batch_labels_filtered = batch_labels[keep_mask]
        batch_boxes_filtered = batch_boxes[keep_mask]
        batch_scores_filtered = batch_scores[keep_mask]
        batch_all_class_scores_filtered = batch_all_class_scores[keep_mask] if batch_all_class_scores is not None else None
        
        # Apply NMS if enabled
        if self.use_nms:
            keep_after_nms = self._apply_nms(batch_boxes_filtered, batch_scores_filtered, batch_labels_filtered)
            batch_labels_filtered = batch_labels_filtered[keep_after_nms]
            batch_boxes_filtered = batch_boxes_filtered[keep_after_nms]
            batch_scores_filtered = batch_scores_filtered[keep_after_nms]
            if batch_all_class_scores_filtered is not None:
                batch_all_class_scores_filtered = batch_all_class_scores_filtered[keep_after_nms]
        
        return {
            "labels": batch_labels_filtered,
            "boxes": batch_boxes_filtered,
            "scores": batch_scores_filtered,
            "all_class_scores": batch_all_class_scores_filtered if batch_all_class_scores_filtered is not None else torch.empty((len(batch_labels_filtered), num_score_classes), dtype=batch_scores.dtype, device=batch_scores.device),
        }

    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        batch_size = logits.shape[0]
        
        # Sanitize NaN/Inf if present (single check)
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Get query padding mask to filter out padded queries
        query_padding_mask = outputs.get("query_padding_mask")
        if query_padding_mask is not None:
            query_padding_mask = query_padding_mask.to(logits.device)
            # query_padding_mask: True = padded, False = valid
            # Mask padded queries by setting their logits to very low values
            # Use -65000 instead of -1e8 for FP16 compatibility (FP16 range: -65504 to 65504)
            logits = logits.masked_fill(query_padding_mask.unsqueeze(-1), -65000.0)
            # Verify masked_fill didn't introduce NaN
            if torch.isnan(logits).any():
                logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        # Convert boxes to xyxy format and scale to original image size
        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        bbox_pred = bbox_pred * orig_target_sizes.repeat(1, 2).unsqueeze(1)
        
        # CRITICAL FIX: Count valid queries per batch before topk selection
        # This ensures we don't select more queries than available valid ones
        if query_padding_mask is not None:
            num_valid_queries_per_batch = (~query_padding_mask).sum(dim=1)  # [B]
        else:
            num_valid_queries_per_batch = torch.full((batch_size,), logits.shape[1], dtype=torch.long, device=logits.device)
        
        # Process scores based on loss type
        if self.use_focal_loss:
            scores = F.sigmoid(logits)  # [B, Q, num_classes]
            
            # Check for NaN/Inf after sigmoid (single check)
            if torch.isnan(scores).any() or torch.isinf(scores).any():
                scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
            
            # CRITICAL FIX: Mask out padding queries before topk to prevent them from being selected
            # Even though logits are masked, ensure scores are also masked for safety
            if query_padding_mask is not None:
                # Set scores of padded queries to 0 (they're already masked in logits, but ensure here too)
                scores = scores.masked_fill(query_padding_mask.unsqueeze(-1), 0.0)
            
            # Flatten scores for topk: [B, Q * num_classes]
            scores_flat = scores.flatten(1)  # [B, Q * num_classes]
            
            # Determine number of queries to select (per batch, respecting valid query count)
            num_topk_per_batch = torch.minimum(
                torch.full((batch_size,), self.num_top_queries, dtype=torch.long, device=scores.device),
                num_valid_queries_per_batch * self.num_classes
            )
            
            # Get topk scores and indices (per batch with different k values)
            # Note: torch.topk doesn't support per-batch k, so we use min of all batches
            num_topk = min(self.num_top_queries, scores.shape[1] * self.num_classes)
            scores_topk, index_flat = torch.topk(scores_flat, num_topk, dim=-1)
            # Extract labels and query indices
            labels_topk = mod(index_flat, self.num_classes)
            query_indices = index_flat // self.num_classes
            
            # Gather boxes for selected queries
            boxes_topk = bbox_pred.gather(
                dim=1, index=query_indices.unsqueeze(-1).expand(-1, -1, bbox_pred.shape[-1])
            )
            # Gather per-class scores for selected queries
            all_class_scores_topk = scores.gather(
                dim=1, index=query_indices.unsqueeze(-1).expand(-1, -1, scores.shape[-1])
            )  # [B, K, num_classes]
        else:
            scores = F.softmax(logits, dim=-1)[:, :, :-1]  # [B, Q, num_classes-1]
            
            # CRITICAL FIX: Mask out padding queries before topk
            if query_padding_mask is not None:
                scores = scores.masked_fill(query_padding_mask.unsqueeze(-1), 0.0)
            
            scores_max, labels = scores.max(dim=-1)  # [B, Q], [B, Q]
            
            num_topk = min(self.num_top_queries, scores_max.shape[1])
            if scores_max.shape[1] > num_topk:
                scores_topk, query_indices = torch.topk(scores_max, num_topk, dim=-1)
                labels_topk = labels.gather(dim=1, index=query_indices)
                boxes_topk = bbox_pred.gather(
                    dim=1, index=query_indices.unsqueeze(-1).expand(-1, -1, bbox_pred.shape[-1])
                )
                all_class_scores_topk = scores.gather(
                    dim=1, index=query_indices.unsqueeze(-1).expand(-1, -1, scores.shape[-1])
                )
            else:
                scores_topk = scores_max
                labels_topk = labels
                boxes_topk = bbox_pred
                all_class_scores_topk = scores
                query_indices = None

        # Process each batch
        results = []
        for b in range(batch_size):
            batch_labels = labels_topk[b]
            batch_boxes = boxes_topk[b]
            batch_scores = scores_topk[b]
            batch_all_class_scores = all_class_scores_topk[b]
            batch_query_indices = query_indices[b] if query_indices is not None else None
            
            result = self._process_batch(
                batch_labels, 
                batch_boxes, 
                batch_scores,
                batch_all_class_scores=batch_all_class_scores,
                query_padding_mask=query_padding_mask[b] if query_padding_mask is not None else None,
                query_indices=batch_query_indices
            )
            results.append(result)
        
        # Handle remap_mscoco_category if needed
        if self.remap_mscoco_category:
            from ...data.dataset import mscoco_label2category
            for result in results:
                if len(result["labels"]) > 0:
                    result["labels"] = torch.tensor(
                        [mscoco_label2category[int(x.item())] for x in result["labels"]],
                        device=result["labels"].device
                    )
        
        # TODO for onnx export
        if self.deploy_mode:
            return ([r["labels"] for r in results], [r["boxes"] for r in results],
                    [r["scores"] for r in results], [r["all_class_scores"] for r in results])
        
        return results

    def deploy(
        self,
    ):
        self.eval()
        self.deploy_mode = True
        return self
