"""
Dome-DETR: Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import math
import sys
from typing import Iterable

import torch
import torch.amp
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from tools.visualize_image_annotation import visualize_detection
from tools.concatenate_images import concatenate_images
import os
import concurrent.futures
import time

from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils, save_samples
from ..optim import ModelEMA, Warmup

SAVE_INTERMEDIATE_VISUALIZE_RESULT = os.environ.get('SAVE_INTERMEDIATE_VISUALIZE_RESULT', 'False') == 'True'

def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    **kwargs,
):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    print_freq = kwargs.get("print_freq", 10)
    writer: SummaryWriter = kwargs.get("writer", None)

    ema: ModelEMA = kwargs.get("ema", None)
    scaler: GradScaler = kwargs.get("scaler", None)
    lr_warmup_scheduler: Warmup = kwargs.get("lr_warmup_scheduler", None)

    output_dir = kwargs.get("output_dir", None)
    num_visualization_batches_per_gpu = kwargs.get("num_visualization_batches_per_gpu", 1)
    num_visualization_samples_per_batch = kwargs.get("num_visualization_samples_per_batch", None)  # Optional: limit samples per batch
    visualization_epoch_interval = kwargs.get("visualization_epoch_interval", 1)  # Save every N epochs
    visualization_save_rank0_only = kwargs.get("visualization_save_rank0_only", False)  # Save only from rank 0

    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        # Save samples for selected epochs (first N batches per GPU)
        # Optionally save only from rank 0 to minimize storage
        rank = dist_utils.get_rank()
        should_save_epoch = (visualization_epoch_interval > 0 and epoch % visualization_epoch_interval == 0)
        should_save_batch = (i < num_visualization_batches_per_gpu)
        should_save_rank = (not visualization_save_rank0_only) or (rank == 0)
        
        if should_save_epoch and should_save_batch and should_save_rank and output_dir is not None:
            # Debug log for first batch of first epoch
            if epoch == 0 and i == 0 and rank == 0:
                print(f"[Visualization] Saving train samples: epoch={epoch}, batch={i}, rank={rank}")
                print(f"  visualization_epoch_interval={visualization_epoch_interval}, should_save_epoch={should_save_epoch}")
                print(f"  num_visualization_batches_per_gpu={num_visualization_batches_per_gpu}, should_save_batch={should_save_batch}")
                print(f"  visualization_save_rank0_only={visualization_save_rank0_only}, should_save_rank={should_save_rank}")
                print(f"  output_dir={output_dir}")
            # Optionally limit number of samples per batch
            if num_visualization_samples_per_batch is not None and num_visualization_samples_per_batch < len(samples):
                samples_to_save = samples[:num_visualization_samples_per_batch]
                targets_to_save = targets[:num_visualization_samples_per_batch]
            else:
                samples_to_save = samples
                targets_to_save = targets
            
            # All ranks save to same folder with rank prefix in filename
            # Note: samples and targets are still on CPU at this point, which is what we want
            # Convert output_dir to string if it's a Path object
            output_dir_str = str(output_dir) if output_dir is not None else None
            save_samples(samples_to_save, targets_to_save, output_dir_str, f"train_samples/train_epoch{epoch:03d}", 
                        normalized=True, box_fmt="cxcywh", rank_prefix=rank)

        no_gt = False
        num_gts = [len(t["labels"]) for t in targets]
        max_gt_num = max(num_gts)
        if max_gt_num == 0: # no gt for denoising will cause error in model forward
            no_gt = True
        samples = samples.to(device)
        # Move targets to device, but skip non-tensor values (e.g., int, bool metadata from MultiCropDatasetWrapper)
        targets = [{k: v.to(device) if hasattr(v, 'to') else v for k, v in t.items()} for t in targets]

        if SAVE_INTERMEDIATE_VISUALIZE_RESULT:
            
            for b, target in enumerate(targets):
                image = samples[b].cpu()
                _, H, W = image.shape
                target_cpu = {}
                for k, v in target.items():
                    if k == 'boxes':
                        target_cpu[k] = v.cpu().detach().clone() * torch.tensor([W, H, W, H])
                    else:
                        target_cpu[k] = v.cpu().detach().clone()
                visualize_detection(image, target_cpu, f"sample_gt", return_image=False, type="xywh")

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)

            if torch.isnan(outputs["pred_boxes"]).any() or torch.isinf(outputs["pred_boxes"]).any():
                print(outputs["pred_boxes"])
                state = model.state_dict()
                new_state = {}
                for key, value in model.state_dict().items():
                    # Replace 'module' with 'model' in each key
                    new_key = key.replace("module.", "")
                    # Add the updated key-value pair to the state dictionary
                    state[new_key] = value
                new_state["model"] = state
                dist_utils.save_on_master(new_state, "./NaN.pth")

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)

            loss: torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        # ema
        if ema is not None:
            ema.update(model)

        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar("Loss/total", loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f"Lr/pg_{j}", pg["lr"], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f"Loss/{k}", v.item(), global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor,
    data_loader,
    coco_evaluator: CocoEvaluator,
    device,
    epoch: int = 0,
    **kwargs,
):
    SAVE_TEST_VISUALIZE_RESULT = os.environ.get('SAVE_TEST_VISUALIZE_RESULT', 'False') == 'True'
    if SAVE_TEST_VISUALIZE_RESULT:
        os.makedirs("visualize_all", exist_ok=True)
        print("Saving visualize results to visualize_all/")
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = "Test:"

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
    
    output_dir = kwargs.get("output_dir", None)
    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)
    
    # For defe Accuracy calculation
    if model.encoder.use_defe:
        total_defe_samples = 0
        ample_defe_predictions = 0
        total_anchor_num = 0

    MAX_PENDING_TASKS = 256
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        pending_futures = []
        
        for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
            global_step = epoch * len(data_loader) + i
            
            # Save validation samples (only from main process, first N batches)
            if global_step < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():
                # Debug log for first batch
                if i == 0:
                    print(f"[Visualization] Saving val samples: epoch={epoch}, batch={i}, global_step={global_step}")
                    print(f"  num_visualization_sample_batch={num_visualization_sample_batch}, output_dir={output_dir}")
                # Note: samples are still on CPU at this point, which is what we want
                # Convert output_dir to string if it's a Path object
                output_dir_str = str(output_dir) if output_dir is not None else None
                save_samples(samples, targets, output_dir_str, "val_samples/val", normalized=False, box_fmt="xyxy")
            
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            image_ids = [t["image_id"].item() for t in targets]
            coco = data_loader.dataset.coco
            file_names = [coco.loadImgs(id)[0]['file_name'] for id in image_ids]

            # In evaluation mode, don't pass targets to model (only needed for training)
            outputs = model(samples, targets=None)
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessor(outputs, orig_target_sizes)

            if SAVE_TEST_VISUALIZE_RESULT:
                process_args = []
                scale_factor = float(samples[0].shape[1] / orig_target_sizes[0][0])
                for i in range(len(targets)):
                    sample_cpu = samples[i].cpu()
                    target_cpu = {k: v.cpu() for k, v in targets[i].items()}
                    result_cpu = {k: v.cpu() for k, v in results[i].items()}
                    process_args.append((
                        sample_cpu,
                        target_cpu,
                        result_cpu,
                        file_names[i],
                        scale_factor
                    ))
                
                if len(pending_futures) >= MAX_PENDING_TASKS:
                    while len(pending_futures) > 0:
                        done_futures = []
                        for future in pending_futures:
                            if future.done():
                                done_futures.append(future)
                        
                        for future in done_futures:
                            pending_futures.remove(future)
                        
                        if not done_futures:
                            time.sleep(0.1)

                for args in process_args:
                    future = executor.submit(process_image_pair, args)
                    pending_futures.append(future)

            res = {target["image_id"].item(): output for target, output in zip(targets, results)}
            if coco_evaluator is not None:
                coco_evaluator.update(res)

            if model.encoder.use_defe:
                # For defe Ample Rate calculation
                pred_defe = outputs['batch_queries_num'][0]
                if pred_defe >= targets[0]['labels'].shape[0]:
                    ample_defe_predictions += 1
                total_defe_samples += 1
                total_anchor_num += outputs['batch_queries_num'][0]

        concurrent.futures.wait(pending_futures)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if model.encoder.use_defe:
        print("defe Ample Rate:", ample_defe_predictions / total_defe_samples)
        print("defe Average Anchor Number:", total_anchor_num / total_defe_samples)

    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if "bbox" in iou_types:
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in iou_types:
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()

    return stats, coco_evaluator


def process_image_pair(args):
    sample, target, result, filename, scale_factor = args
    sample_img = visualize_detection(sample, target, f"sample_{filename}", return_image=True)
    result_img = visualize_detection(sample, result, f"result_{filename}", 
                                   scale_factor=scale_factor, return_image=True)
    concatenate_images(sample_img, result_img, output_path=f"visualize_all/{filename}")