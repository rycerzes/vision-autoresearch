"""Object-detection slice for the shared HF vision runner.

This is stable infrastructure - do NOT edit during experiments.
Experiments modify config YAMLs only.

Adapted from huggingface/skills huggingface-vision-trainer.
"""

# pyright: reportPrivateImportUsage=false
# PyTorch typings mark many public APIs (e.g. torch.tensor) as private re-exports.

import logging
import math
import os
import re
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, cast

import albumentations as A
import numpy as np
import torch
import transformers
from datasets import load_dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import (
    AutoImageProcessor,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.image_processing_base import BatchFeature
from transformers.image_transforms import center_to_corners_format
from transformers.trainer_utils import EvalPrediction

from vision_lab.hf_vision.adaptation import apply_adaptation_mode
from vision_lab.hf_vision.loaders import load_hf_vision_model
from vision_lab.hf_vision.runner_session import finish_trackio_session, setup_hf_training_environment
from vision_lab.hf_vision.summary_block import print_vision_autoresearch_summary

logger = logging.getLogger(__name__)

# Stateful eval metric when TrainingArguments.batch_eval_metrics is True (see Trainer.evaluation_loop).
_eval_map_metric: MeanAveragePrecision | None = None


# Helpers

@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


def format_image_annotations_as_coco(
    image_id: str | int, categories: list[int], areas: list[float], bboxes: list[tuple[float]]
) -> dict[str, Any]:
    """Build one image's COCO-style annotation dict for RT-DETR / DFine processors (image_id must be int)."""
    iid = int(image_id) if isinstance(image_id, (int, np.integer)) else int(str(image_id))
    annotations = []
    for category, area, bbox in zip(categories, areas, bboxes):
        formatted_annotation = {
            "image_id": iid,
            "category_id": category,
            "iscrowd": 0,
            "area": area,
            "bbox": list(bbox),
        }
        annotations.append(formatted_annotation)
    return {"image_id": iid, "annotations": annotations}


def detect_bbox_format_from_samples(dataset, image_col="image", objects_col="objects", num_samples=50):
    """Detect whether bboxes are xyxy (Pascal VOC) or xywh (COCO)."""
    exceeds_if_xywh = 0
    exceeds_if_xyxy = 0
    total = 0

    for example in dataset.select(range(min(num_samples, len(dataset)))):
        img_w, img_h = example[image_col].size
        for bbox in example[objects_col]["bbox"]:
            if len(bbox) != 4:
                continue
            a, b, c, d = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            total += 1
            if c < a or d < b:
                return "xywh"
            if a + c > img_w * 1.05:
                exceeds_if_xywh += 1
            if b + d > img_h * 1.05:
                exceeds_if_xywh += 1
            if c > img_w * 1.05:
                exceeds_if_xyxy += 1
            if d > img_h * 1.05:
                exceeds_if_xyxy += 1

    if total == 0:
        return "xywh"
    fmt = "xyxy" if exceeds_if_xywh > exceeds_if_xyxy else "xywh"
    logger.info(f"Detected bbox format: {fmt} (checked {total} bboxes from {min(num_samples, len(dataset))} images)")
    return fmt


def sanitize_dataset(dataset, bbox_format="xywh", image_col="image", objects_col="objects"):
    """Validate bboxes, convert xyxy->xywh if needed, clip to image bounds, remove degenerate entries."""
    convert_xyxy = bbox_format == "xyxy"

    def _validate(example):
        img_w, img_h = example[image_col].size
        objects = example[objects_col]
        bboxes = objects["bbox"]
        n = len(bboxes)
        valid_indices = []
        converted_bboxes = []

        for i, bbox in enumerate(bboxes):
            if len(bbox) != 4:
                continue
            vals = [float(v) for v in bbox]
            if not all(math.isfinite(v) for v in vals):
                continue
            if convert_xyxy:
                x_min, y_min, x_max, y_max = vals
                w, h = x_max - x_min, y_max - y_min
            else:
                x_min, y_min, w, h = vals
            if w <= 0 or h <= 0:
                continue
            x_min, y_min = max(0.0, x_min), max(0.0, y_min)
            if x_min >= img_w or y_min >= img_h:
                continue
            w = min(w, img_w - x_min)
            h = min(h, img_h - y_min)
            if w * h < 1.0:
                continue
            valid_indices.append(i)
            converted_bboxes.append([x_min, y_min, w, h])

        new_objects = {}
        for key, value in objects.items():
            if key == "bbox":
                new_objects["bbox"] = converted_bboxes
            elif isinstance(value, list) and len(value) == n:
                new_objects[key] = [value[j] for j in valid_indices]
            else:
                new_objects[key] = value

        if "area" not in new_objects or len(new_objects.get("area", [])) != len(converted_bboxes):
            new_objects["area"] = [b[2] * b[3] for b in converted_bboxes]
        example[objects_col] = new_objects
        return example

    before = len(dataset)
    dataset = dataset.map(_validate)
    dataset = dataset.filter(lambda ex: len(ex[objects_col]["bbox"]) > 0)
    after = len(dataset)
    if before != after:
        logger.warning(f"Dropped {before - after}/{before} images with no valid bboxes after sanitization")
    logger.info(f"Bbox sanitization complete: {after} images with valid bboxes remain")
    return dataset


def convert_bbox_yolo_to_pascal(boxes: torch.Tensor, image_size: torch.Tensor | Sequence[int] | np.ndarray) -> torch.Tensor:
    boxes = cast(torch.Tensor, center_to_corners_format(cast(Any, boxes)))
    if isinstance(image_size, torch.Tensor):
        flat = image_size.detach().cpu().reshape(-1).tolist()
    elif isinstance(image_size, np.ndarray):
        flat = np.asarray(image_size).reshape(-1).tolist()
    else:
        flat = list(image_size)
    if len(flat) >= 2:
        height, width = int(flat[0]), int(flat[1])
    elif len(flat) == 1:
        side = int(flat[0])
        height, width = side, side
    else:
        raise ValueError(f"image_size must include height and width (got {flat!r})")
    boxes = boxes * torch.tensor([[width, height, width, height]])
    return boxes


def augment_and_transform_batch(
    examples: Mapping[str, Any],
    transform: A.Compose,
    image_processor: AutoImageProcessor,
    return_pixel_mask: bool = False,
) -> BatchFeature:
    images = []
    annotations = []
    image_ids = examples["image_id"] if "image_id" in examples else range(len(examples["image"]))
    for image_id, image, objects in zip(image_ids, examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))
        bboxes = objects["bbox"]
        categories = objects["category"]
        areas = objects["area"]
        valid = [
            (b, c, a)
            for b, c, a in zip(bboxes, categories, areas)
            if len(b) == 4 and b[2] > 0 and b[3] > 0 and b[0] >= 0 and b[1] >= 0
        ]
        if valid:
            bboxes, categories, areas = zip(*valid)
        else:
            bboxes, categories, areas = [], [], []

        output = transform(image=image, bboxes=list(bboxes), category=list(categories))
        images.append(output["image"])
        post_areas = [b[2] * b[3] for b in output["bboxes"]] if output["bboxes"] else []
        formatted_annotations = format_image_annotations_as_coco(
            image_id, output["category"], post_areas, output["bboxes"]
        )
        annotations.append(formatted_annotations)

    processor_call = cast(Any, image_processor)
    result = processor_call(images=images, annotations=annotations, return_tensors="pt")
    if not return_pixel_mask:
        result.pop("pixel_mask", None)
    return result


def collate_fn(batch: list[BatchFeature]) -> dict[str, torch.Tensor | list[Any]]:
    data: dict[str, torch.Tensor | list[Any]] = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    if "pixel_mask" in batch[0]:
        data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
    return data


def _as_torch(t: Any) -> torch.Tensor:
    if isinstance(t, torch.Tensor):
        return t.detach().cpu()
    return torch.tensor(t)


def _hw_pair_tensor(orig: Any) -> torch.Tensor:
    """Return shape ``[2]`` tensor ``(height, width)`` for RT-DETR ``target_sizes`` / box scaling."""
    t = _as_torch(orig).reshape(-1)
    if t.numel() >= 2:
        return t[:2].clone()
    if t.numel() == 1:
        return t.expand(2).clone()
    raise ValueError(f"orig_size must be non-empty (got {orig!r})")


def _flatten_prediction_leaves(pred: Any, out: list[Any]) -> None:
    """DFine / DETR may nest auxiliary dicts inside the ``prediction_step`` logits structure."""
    if isinstance(pred, dict):
        for v in pred.values():
            _flatten_prediction_leaves(v, out)
    elif isinstance(pred, (list, tuple)):
        for x in pred:
            _flatten_prediction_leaves(x, out)
    elif isinstance(pred, np.ndarray) and pred.dtype == object:
        for x in pred.tolist():
            _flatten_prediction_leaves(x, out)
    else:
        out.append(pred)


def _unpack_logits_boxes(predictions: Any, num_labels: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Split DFine / RT-DETR outputs into logits [B,Q,C] and pred_boxes [B,Q,4].

    Auxiliary decoder heads may expose many extra tensors; prefer the head whose class
    dimension matches ``num_labels`` (when known) and whose query count matches the
    primary box prediction tensor.
    """
    if predictions is None:
        raise ValueError("predictions is None")
    leaves: list[Any] = []
    _flatten_prediction_leaves(predictions, leaves)
    tensors_3d: list[torch.Tensor] = []
    for x in leaves:
        try:
            t = _as_torch(x)
        except (TypeError, RuntimeError):
            continue
        if t.dim() == 3:
            tensors_3d.append(t)
    boxes_candidates = [t for t in tensors_3d if t.shape[-1] == 4]
    if len(boxes_candidates) < 1:
        raise ValueError(f"No box tensor (last_dim==4) in predictions {predictions!r}")
    boxes_t = max(boxes_candidates, key=lambda t: t.shape[1])
    b, q, _ = boxes_t.shape
    logits_matches = [
        t for t in tensors_3d if t.shape[-1] != 4 and t.shape[0] == b and t.shape[1] == q
    ]
    if num_labels is not None:
        capped = [t for t in logits_matches if t.shape[-1] <= num_labels + 2]
        if capped:
            logits_matches = capped
    if not logits_matches:
        raise ValueError(f"No logits tensor matching boxes {(b, q)} in predictions {predictions!r}")
    logits_t = min(logits_matches, key=lambda t: t.shape[-1])
    return logits_t, boxes_t


def _per_image_label_dicts(label_ids: Any) -> list[dict[str, Any]]:
    """Turn Trainer eval `label_ids` into one HF labels dict per image (CPU tensors)."""
    if label_ids is None:
        return []
    if isinstance(label_ids, np.ndarray):
        if label_ids.dtype == object:
            label_ids = label_ids.tolist()
        else:
            raise TypeError(f"Unexpected label_ids ndarray dtype {label_ids.dtype}")
    pad_token = -100
    if isinstance(label_ids, dict):
        boxes_b = _as_torch(label_ids["boxes"])
        classes_b = _as_torch(label_ids["class_labels"])
        if boxes_b.dim() != 3:
            raise TypeError(f"Expected batched boxes [B,N,4], got shape {tuple(boxes_b.shape)}")
        bsz = boxes_b.shape[0]
        ob = _as_torch(label_ids["orig_size"])
        if ob.dim() == 1:
            if ob.numel() == 2:
                orig_rows = ob.unsqueeze(0).expand(bsz, -1).clone()
            elif ob.numel() == 2 * bsz:
                orig_rows = ob.reshape(bsz, 2)
            else:
                raise ValueError(f"Unexpected 1D orig_size length {ob.numel()} for batch {bsz}")
        elif ob.dim() == 2:
            if ob.shape[0] == bsz:
                orig_rows = ob[:, :2]
            elif ob.shape[1] == bsz:
                orig_rows = ob[:2, :].transpose(0, 1).contiguous()
            else:
                orig_rows = ob.reshape(bsz, -1)[:, :2]
        else:
            orig_rows = ob.reshape(bsz, -1)[:, :2]

        out: list[dict[str, Any]] = []
        for i in range(bsz):
            cl = classes_b[i]
            bx = boxes_b[i]
            valid = cl.ne(pad_token)
            if valid.any():
                bx = bx[valid]
                cl = cl[valid]
            else:
                bx = bx.new_zeros((0, 4))
                cl = cl.new_zeros((0,), dtype=cl.dtype)
            orig = orig_rows[i].reshape(-1)[:2]
            out.append({"boxes": bx, "class_labels": cl.long(), "orig_size": orig})
        return out
    if isinstance(label_ids, (list, tuple)):
        return [cast(dict[str, Any], x) for x in label_ids]
    raise TypeError(f"Unexpected label_ids type {type(label_ids)!r}")


@torch.no_grad()
def compute_metrics(
    evaluation_results: EvalPrediction,
    image_processor: AutoImageProcessor,
    threshold: float = 0.0,
    id2label: Mapping[int, str] | None = None,
    num_labels: int | None = None,
    *,
    compute_result: bool = False,
) -> Mapping[str, float]:
    """Detection mAP; supports Transformers ``batch_eval_metrics`` (accumulate until ``compute_result``)."""
    global _eval_map_metric

    predictions = evaluation_results.predictions
    label_ids = evaluation_results.label_ids

    if predictions is None or label_ids is None:
        if compute_result and _eval_map_metric is not None:
            metrics = _eval_map_metric.compute()
            _eval_map_metric = None
            return _format_detection_metrics(metrics, id2label)
        return {}

    batch_logits, batch_boxes = _unpack_logits_boxes(predictions, num_labels=num_labels)
    image_entries = _per_image_label_dicts(label_ids)
    if batch_logits.shape[0] != len(image_entries):
        raise ValueError(
            f"Batch size mismatch: logits batch {batch_logits.shape[0]} vs labels {len(image_entries)}"
        )

    target_sizes = torch.stack([_hw_pair_tensor(x["orig_size"]) for x in image_entries])
    post_processed_targets = []
    for image_target in image_entries:
        boxes = _as_torch(image_target["boxes"])
        boxes = convert_bbox_yolo_to_pascal(boxes, _hw_pair_tensor(image_target["orig_size"]))
        labels = _as_torch(image_target["class_labels"]).long()
        post_processed_targets.append({"boxes": boxes, "labels": labels})

    output = ModelOutput(logits=batch_logits, pred_boxes=batch_boxes)
    post_processed_predictions = cast(Any, image_processor).post_process_object_detection(
        output, threshold=threshold, target_sizes=target_sizes
    )

    if _eval_map_metric is None:
        _eval_map_metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
        _eval_map_metric.warn_on_many_detections = False
    _eval_map_metric.update(post_processed_predictions, post_processed_targets)

    if not compute_result:
        return {}

    metrics = _eval_map_metric.compute()
    _eval_map_metric = None
    return _format_detection_metrics(metrics, id2label)


def _format_detection_metrics(
    metrics: dict[str, torch.Tensor], id2label: Mapping[int, str] | None
) -> dict[str, float]:
    classes = metrics.pop("classes", None)
    map_per_class = metrics.pop("map_per_class", None)
    mar_pc_key = None
    mar_candidates = [k for k in metrics if re.fullmatch(r"mar_\d+_per_class", k)]
    if mar_candidates:
        mar_pc_key = max(mar_candidates, key=lambda k: int(k.split("_")[1]))
    mar_per_class = metrics.pop(mar_pc_key, None) if mar_pc_key else None
    mar_suffix = mar_pc_key.split("_")[1] if mar_pc_key else "100"

    if (
        classes is not None
        and map_per_class is not None
        and mar_per_class is not None
        and classes.numel() > 0
    ):
        if classes.dim() == 0:
            classes = classes.unsqueeze(0)
            map_per_class = map_per_class.unsqueeze(0)
            mar_per_class = mar_per_class.unsqueeze(0)
        for class_id, class_map, class_mar in zip(classes, map_per_class, mar_per_class):
            cid = int(class_id.item())
            if id2label is not None:
                class_name = id2label.get(cid, f"class_{cid}")
            else:
                class_name = cid
            metrics[f"map_{class_name}"] = class_map
            metrics[f"mar_{mar_suffix}_{class_name}"] = class_mar

    out: dict[str, float] = {}
    for k, v in metrics.items():
        out[k] = round(v.item(), 4) if isinstance(v, torch.Tensor) else round(float(v), 4)
    return out


# CLI dataclasses

@dataclass
class DataTrainingArguments:
    dataset_name: str = field(
        default="cppe-5",
        metadata={"help": "Name of a dataset from the Hub."},
    )
    dataset_config_name: str | None = field(
        default=None,
        metadata={"help": "The configuration name of the dataset."},
    )
    train_val_split: float | None = field(
        default=0.15,
        metadata={"help": "Fraction to split off of train for validation."},
    )
    image_square_size: int | None = field(
        default=640,
        metadata={"help": "Resize longest edge to this value, pad to square."},
    )
    max_train_samples: int | None = field(
        default=None,
        metadata={"help": "Truncate training set (for debugging)."},
    )
    max_eval_samples: int | None = field(
        default=None,
        metadata={"help": "Truncate evaluation set."},
    )
    use_fast: bool | None = field(
        default=True,
        metadata={"help": "Use fast torchvision-based image processor."},
    )


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="ustc-community/dfine-small-coco",
        metadata={"help": "Pretrained model identifier."},
    )
    config_name: str | None = field(
        default=None,
        metadata={"help": "Pretrained config name or path."},
    )
    cache_dir: str | None = field(
        default=None,
        metadata={"help": "Cache directory for pretrained models."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "Model version (branch, tag, or commit)."},
    )
    image_processor_name: str | None = field(
        default=None,
        metadata={"help": "Name or path of image processor config."},
    )
    ignore_mismatched_sizes: bool = field(
        default=True,
        metadata={"help": "Allow loading weights when num_labels differs."},
    )
    token: str | None = field(
        default=None,
        metadata={"help": "Auth token for private models/datasets."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Trust remote code from Hub repos."},
    )
    model_loader: str = field(
        default="auto_task_head",
        metadata={"help": "Weight graph: auto_task_head (AutoModelForObjectDetection) only for detect."},
    )
    adaptation_mode: str = field(
        default="full_finetune",
        metadata={"help": "Training posture (see vision_lab.hf_vision.constants.ADAPTATION_MODE_CHOICES)."},
    )


# Main

def main():
    if len(sys.argv) != 2 or not sys.argv[1].endswith((".yaml", ".yml", ".json")):
        raise SystemExit("detect_train is an internal runner slice; use train_hf_vision.py <config.yaml|json>.")
    run_from_config(Path(os.path.abspath(sys.argv[1])))


def run_from_config(config_path: Path) -> None:
    start_time = time.time()

    parser = HfArgumentParser(cast(Any, (ModelArguments, DataTrainingArguments, TrainingArguments)))

    if config_path.suffix.lower() in (".yaml", ".yml"):
        model_args, data_args, training_args = parser.parse_yaml_file(
            yaml_file=str(config_path), allow_extra_keys=True
        )
    elif config_path.suffix.lower() == ".json":
        model_args, data_args, training_args = parser.parse_json_file(json_file=str(config_path))
    else:
        raise SystemExit("Config must be .yaml, .yml, or .json")

    setup_hf_training_environment(training_args, logger=logger)

    logger.info(
        "HF vision runner (detect vertical): model_loader=%s adaptation_mode=%s",
        model_args.model_loader.strip(),
        model_args.adaptation_mode.strip(),
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Load dataset
    dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Bbox sanitization
    bbox_format = detect_bbox_format_from_samples(dataset["train"])
    if bbox_format == "xyxy":
        logger.info("Converting bboxes from xyxy (Pascal VOC) -> xywh (COCO) format")
    for split_name in list(dataset.keys()):
        dataset[split_name] = sanitize_dataset(dataset[split_name], bbox_format=bbox_format)

    for split_name in list(dataset.keys()):
        ds_split = dataset[split_name]
        # HF detection datasets often ship string image_id; RT-DETR image processors require int ids.
        if "image_id" in ds_split.column_names:
            ds_split = ds_split.remove_columns(["image_id"])
        dataset[split_name] = ds_split.add_column("image_id", list(range(len(ds_split))))

    dataset["train"] = dataset["train"].shuffle(seed=training_args.seed)

    # Train/val split
    data_args.train_val_split = None if "validation" in dataset else data_args.train_val_split
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = dataset["train"].train_test_split(data_args.train_val_split, seed=training_args.seed)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    # Discover categories
    categories = None
    try:
        if isinstance(dataset["train"].features["objects"], dict):
            cat_feature = dataset["train"].features["objects"]["category"].feature
        else:
            cat_feature = dataset["train"].features["objects"].feature["category"]
        if hasattr(cat_feature, "names"):
            categories = cat_feature.names
    except (AttributeError, KeyError):
        pass

    if categories is None:
        logger.info("Category feature is not ClassLabel -- scanning dataset to discover labels...")
        unique_cats = set()
        for raw_row in dataset["train"]:
            example = cast(dict[str, Any], raw_row)
            cats = example["objects"]["category"]
            if isinstance(cats, list):
                unique_cats.update(cats)
            else:
                unique_cats.add(cats)
        if all(isinstance(c, int) for c in unique_cats):
            max_cat = max(unique_cats)
            categories = [f"class_{i}" for i in range(max_cat + 1)]
        elif all(isinstance(c, str) for c in unique_cats):
            categories = sorted(unique_cats)
        else:
            categories = [str(c) for c in sorted(unique_cats, key=str)]
        logger.info(f"Discovered {len(categories)} categories: {categories}")

    id2label = dict(enumerate(categories))
    label2id = {v: k for k, v in id2label.items()}

    # Remap string categories to integer IDs if needed
    sample_cats = dataset["train"][0]["objects"]["category"]
    if sample_cats and isinstance(sample_cats[0], str):
        logger.info(f"Remapping string categories to integer IDs: {label2id}")

        def _remap_categories(example):
            objects = example["objects"]
            objects["category"] = [label2id[c] for c in objects["category"]]
            example["objects"] = objects
            return example

        for split_name in list(dataset.keys()):
            dataset[split_name] = dataset[split_name].map(_remap_categories)

    # Truncate
    if data_args.max_train_samples is not None:
        max_train = min(data_args.max_train_samples, len(dataset["train"]))
        dataset["train"] = dataset["train"].select(range(max_train))
    if data_args.max_eval_samples is not None and "validation" in dataset:
        max_eval = min(data_args.max_eval_samples, len(dataset["validation"]))
        dataset["validation"] = dataset["validation"].select(range(max_eval))

    ml = model_args.model_loader.strip()
    model, image_processor = load_hf_vision_model(
        task_type="detect",
        model_loader=ml,
        model_name_or_path=model_args.model_name_or_path,
        config_name=model_args.config_name,
        num_labels=len(categories),
        label2id=label2id,
        id2label=id2label,
        cache_dir=model_args.cache_dir,
        model_revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        image_processor_name=model_args.image_processor_name,
    )
    if data_args.image_square_size is not None:
        image_processor.size = {
            "max_height": data_args.image_square_size,
            "max_width": data_args.image_square_size,
        }
        image_processor.pad_size = {
            "height": data_args.image_square_size,
            "width": data_args.image_square_size,
        }
    if data_args.use_fast is not None and hasattr(image_processor, "use_fast"):
        image_processor.use_fast = data_args.use_fast
    apply_adaptation_mode(model, model_args.adaptation_mode, architecture="detect")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("After adaptation_mode=%s: %s/%s params trainable", model_args.adaptation_mode, trainable, total)

    # Augmentation
    max_size = data_args.image_square_size
    train_augment_and_transform = A.Compose(
        [
            # Always produce a fixed square before lighter augmentations. If random crop is skipped too often,
            # inputs stay variable-sized → RT-DETR/DFine preprocess stacks / backbone strides break (OOM/off-by-one).
            A.SmallestMaxSize(max_size=max_size, p=1.0),
            A.RandomSizedBBoxSafeCrop(height=max_size, width=max_size, p=1.0),
            A.OneOf(
                [
                    A.Blur(blur_limit=7, p=0.5),
                    A.MotionBlur(blur_limit=7, p=0.5),
                    A.Defocus(radius=(1, 5), alias_blur=(0.1, 0.25), p=0.1),
                ],
                p=0.1,
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.1),
            # Guarantee fixed `max_size` square for batched RT-DETR preprocess (Perspective/Perspective-like ops break stack).
            A.Resize(height=max_size, width=max_size, p=1.0),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=25),
    )
    validation_transform = A.Compose(
        [
            A.LongestMaxSize(max_size=max_size, p=1.0),
            A.PadIfNeeded(
                min_height=max_size,
                min_width=max_size,
                border_mode=0,
                value=(0, 0, 0),
            ),
            A.CenterCrop(height=max_size, width=max_size, p=1.0),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=25),
    )

    train_transform_batch = partial(
        augment_and_transform_batch, transform=train_augment_and_transform, image_processor=image_processor
    )
    validation_transform_batch = partial(
        augment_and_transform_batch, transform=validation_transform, image_processor=image_processor
    )

    dataset["train"] = dataset["train"].with_transform(train_transform_batch)
    eval_split = "validation" if "validation" in dataset else "test"
    dataset[eval_split] = dataset[eval_split].with_transform(validation_transform_batch)
    if "test" in dataset and eval_split != "test":
        dataset["test"] = dataset["test"].with_transform(validation_transform_batch)

    def eval_compute_metrics_fn(eval_pred: EvalPrediction, compute_result: bool = False) -> dict[str, float]:
        return dict(
            compute_metrics(
                eval_pred,
                image_processor=image_processor,
                threshold=0.0,
                id2label=id2label,
                num_labels=getattr(model.config, "num_labels", None),
                compute_result=compute_result,
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset[eval_split] if training_args.do_eval else None,
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )

    # Train
    train_metrics = {}
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        train_metrics = train_result.metrics
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)
        trainer.save_state()

    # Evaluate
    eval_metrics = {}
    if training_args.do_eval:
        test_dataset = dataset.get("test", dataset.get("validation"))
        if test_dataset is None:
            raise RuntimeError("do_eval is True but no validation/test split exists")
        test_prefix = "test" if "test" in dataset else "eval"
        eval_metrics = trainer.evaluate(eval_dataset=cast(Any, test_dataset), metric_key_prefix=test_prefix)
        trainer.log_metrics(test_prefix, eval_metrics)
        trainer.save_metrics(test_prefix, eval_metrics)

    training_seconds = time.time() - start_time
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0

    finish_trackio_session()

    print_vision_autoresearch_summary(
        "detect", eval_metrics, train_metrics, training_seconds, peak_vram_mb
    )

    # Push to Hub
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": data_args.dataset_name,
        "tags": ["object-detection", "vision"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


