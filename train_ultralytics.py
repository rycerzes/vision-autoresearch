"""Ultralytics YOLO training for multiple CV tasks using Hugging Face Hub datasets.

Stable infrastructure — experiments modify config YAMLs only.

Supported ``task_type`` values (promotion metric in parentheses):
  detect_yolo (mAP), track_yolo (mAP), segment_yolo (IoU mask mAP proxy),
  classify_yolo (accuracy), pose_yolo (mAP pose proxy), obb_yolo (mAP OBB proxy).

``track_yolo`` trains a detector (Ultralytics has no ``train`` mode for tracking);
use the same detection dataset contract; tracking inference is separate.
"""

from __future__ import annotations

import csv
import logging
import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import yaml
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import login
from transformers import HfArgumentParser, TrainingArguments

import trackio
from train_detect import ModelArguments, detect_bbox_format_from_samples, sanitize_dataset
from ultralytics import YOLO

logger = logging.getLogger(__name__)

LEDGER_TASKS = frozenset(
    {
        "detect_yolo",
        "track_yolo",
        "segment_yolo",
        "classify_yolo",
        "pose_yolo",
        "obb_yolo",
    }
)


@dataclass
class YoloDataArguments:
    dataset_name: str = field(default="cppe-5", metadata={"help": "HF Hub dataset name."})
    dataset_config_name: Optional[str] = field(default=None, metadata={"help": "Dataset config name."})
    train_val_split: Optional[float] = field(
        default=0.15,
        metadata={"help": "Holdout fraction from train when no validation split exists."},
    )
    image_square_size: int = field(default=640, metadata={"help": "YOLO imgsz."})
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "Cap training samples."})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "Cap validation samples."})
    label_column: str = field(
        default="label",
        metadata={"help": "Classification label column (classify_yolo)."},
    )
    mask_column: Optional[str] = field(
        default=None,
        metadata={"help": "Semantic mask column for segment_yolo (auto if unset)."},
    )


def _slug_class_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name)).strip("_")
    return slug[:120] if slug else "class"


def read_train_metrics_from_csv(run_dir: Path) -> dict[str, Any]:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return {"train_loss": 0.0, "epoch": 0}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"train_loss": 0.0, "epoch": 0}
    last = rows[-1]
    train_loss = 0.0
    for key in last:
        if not key:
            continue
        lk = key.lower()
        if "train" in lk and "loss" in lk:
            try:
                train_loss = float(last[key])
                break
            except (TypeError, ValueError):
                continue
    epoch = 0
    if "epoch" in last:
        try:
            epoch = int(float(last["epoch"]))
        except (TypeError, ValueError):
            epoch = 0
    return {"train_loss": train_loss, "epoch": epoch}


def pick_csv_metric(last_row: dict[str, str], keys: tuple[str, ...]) -> float:
    for k in keys:
        raw = last_row.get(k)
        if raw in (None, ""):
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return 0.0


def read_detect_metrics(run_dir: Path) -> dict[str, float]:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return {"mAP": 0.0, "mAP_50": 0.0, "mAR": 0.0}
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    if not rows:
        return {"mAP": 0.0, "mAP_50": 0.0, "mAR": 0.0}
    last = rows[-1]
    return {
        "mAP": pick_csv_metric(
            last,
            ("metrics/mAP50-95(B)", "metrics/mAP50-95(M)", "metrics/mAP50-95(P)"),
        ),
        "mAP_50": pick_csv_metric(
            last,
            ("metrics/mAP50(B)", "metrics/mAP50(M)", "metrics/mAP50(P)"),
        ),
        "mAR": pick_csv_metric(
            last,
            ("metrics/recall(B)", "metrics/recall(M)", "metrics/recall(P)"),
        ),
    }


def read_classify_metrics(run_dir: Path) -> dict[str, float]:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return {"accuracy": 0.0}
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    if not rows:
        return {"accuracy": 0.0}
    last = rows[-1]
    acc = pick_csv_metric(
        last,
        (
            "metrics/accuracy_top1",
            "metrics/accuracy_top1(top1)",
            "accuracy/top1",
        ),
    )
    return {"accuracy": acc}


def read_segment_metrics(run_dir: Path) -> dict[str, float]:
    """Use mask branch mAP@50 as IoU-like promotion proxy for segment_yolo."""
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return {"iou": 0.0, "mAP": 0.0, "mAP_50": 0.0}
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    if not rows:
        return {"iou": 0.0, "mAP": 0.0, "mAP_50": 0.0}
    last = rows[-1]
    map50_m = pick_csv_metric(last, ("metrics/mAP50(M)",))
    map5095_m = pick_csv_metric(last, ("metrics/mAP50-95(M)",))
    return {"iou": map50_m, "mAP": map5095_m, "mAP_50": map50_m}


def emit_summary(
    ledger_task: str,
    metrics_lines: dict[str, float],
    train_metrics: dict[str, Any],
    training_seconds: float,
    peak_vram_mb: float,
) -> None:
    print("\n--- VISION AUTORESEARCH SUMMARY ---")
    print(f"task_type: {ledger_task}")
    order = ("mAP", "mAP_50", "mAR", "accuracy", "iou", "dice")
    for key in order:
        if key in metrics_lines:
            print(f"{key}: {metrics_lines[key]}")
    for key, val in sorted(metrics_lines.items()):
        if key not in order:
            print(f"{key}: {val}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"peak_vram_mb: {peak_vram_mb:.0f}")
    print(f"train_loss: {train_metrics.get('train_loss', 0.0)}")
    print(f"num_train_epochs: {train_metrics.get('epoch', 0)}")
    print("--- END SUMMARY ---")


def discover_categories_and_remap(dataset: DatasetDict) -> dict[int, str]:
    categories = None
    try:
        if isinstance(dataset["train"].features["objects"], dict):
            cat_feature = dataset["train"].features["objects"]["category"].feature
        else:
            cat_feature = dataset["train"].features["objects"].feature["category"]
        if hasattr(cat_feature, "names"):
            categories = cat_feature.names
    except (AttributeError, KeyError, TypeError):
        pass

    if categories is None:
        logger.info("Category feature is not ClassLabel — scanning dataset...")
        unique_cats: set[Any] = set()
        for example in dataset["train"]:
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

    id2label = dict(enumerate(categories))
    label2id = {v: k for k, v in id2label.items()}

    sample_cats = dataset["train"][0]["objects"]["category"]
    if sample_cats and isinstance(sample_cats[0], str):

        def _remap(example: dict[str, Any]) -> dict[str, Any]:
            objects = example["objects"]
            objects["category"] = [label2id[c] for c in objects["category"]]
            example["objects"] = objects
            return example

        for split_name in list(dataset.keys()):
            dataset[split_name] = dataset[split_name].map(_remap)

    return id2label


def bbox_to_yolo_line(bbox: list[float], cat_id: int, img_w: int, img_h: int) -> str | None:
    if len(bbox) != 4:
        return None
    x, y, w, h = (float(v) for v in bbox)
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        return None
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    if nw <= 0 or nh <= 0:
        return None
    return f"{int(cat_id)} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def prepare_detection_like_dataset(
    model_args: ModelArguments,
    data_args: YoloDataArguments,
    training_args: TrainingArguments,
) -> tuple[DatasetDict, dict[int, str]]:
    dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    bbox_format = detect_bbox_format_from_samples(dataset["train"])
    if bbox_format == "xyxy":
        logger.info("Converting bboxes from xyxy → xywh before YOLO export")
    for split_name in list(dataset.keys()):
        dataset[split_name] = sanitize_dataset(dataset[split_name], bbox_format=bbox_format)

    for split_name in list(dataset.keys()):
        if "image_id" not in dataset[split_name].column_names:
            dataset[split_name] = dataset[split_name].add_column(
                "image_id", list(range(len(dataset[split_name])))
            )

    dataset["train"] = dataset["train"].shuffle(seed=training_args.seed)

    tv_split = None if "validation" in dataset else data_args.train_val_split
    if isinstance(tv_split, float) and tv_split > 0.0:
        split = dataset["train"].train_test_split(tv_split, seed=training_args.seed)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    id2label = discover_categories_and_remap(dataset)

    if data_args.max_train_samples is not None:
        max_train = min(data_args.max_train_samples, len(dataset["train"]))
        dataset["train"] = dataset["train"].select(range(max_train))

    val_key = "validation" if "validation" in dataset else "test"
    if val_key not in dataset:
        raise SystemExit(
            "Dataset has no validation or test split; use a dataset with val/test "
            "or enable train_val_split."
        )

    if data_args.max_eval_samples is not None:
        max_ev = min(data_args.max_eval_samples, len(dataset[val_key]))
        dataset[val_key] = dataset[val_key].select(range(max_ev))

    return dataset, id2label


def export_split_detect(
    split_ds: Dataset,
    images_dir: Path,
    labels_dir: Path,
    prefix: str,
    line_builder: Callable[[dict[str, Any], int, int], list[str]],
) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(len(split_ds)):
        example = split_ds[idx]
        image = example["image"]
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        img_w, img_h = image.size
        stem = f"{prefix}_{idx:06d}"
        image.save(images_dir / f"{stem}.jpg", format="JPEG", quality=95)
        lines = line_builder(example, img_w, img_h)
        (labels_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")


def detect_label_lines(example: dict[str, Any], img_w: int, img_h: int) -> list[str]:
    objects = example["objects"]
    bboxes = objects["bbox"]
    cats = objects["category"]
    lines: list[str] = []
    for bbox, cat in zip(bboxes, cats):
        line = bbox_to_yolo_line(list(bbox), int(cat), img_w, img_h)
        if line:
            lines.append(line)
    return lines


def pose_label_lines(example: dict[str, Any], img_w: int, img_h: int) -> list[str]:
    objects = example["objects"]
    bboxes = objects["bbox"]
    cats = objects["category"]
    kpts_all = objects["keypoints"]
    lines: list[str] = []
    for bbox, cat, kpts in zip(bboxes, cats, kpts_all):
        det = bbox_to_yolo_line(list(bbox), int(cat), img_w, img_h)
        if not det:
            continue
        parts = det.split()
        cls = parts[0]
        cx, cy, w, h = map(float, parts[1:])
        kflat = list(kpts)
        nk = len(kflat)
        if nk % 3 != 0 and nk % 2 == 0:
            kflat = []
            for i in range(0, len(list(kpts)), 2):
                x, y = float(kpts[i]), float(kpts[i + 1])
                kflat.extend([x, y, 2.0])
        triplets: list[str] = []
        for i in range(0, len(kflat), 3):
            x, y, v = float(kflat[i]), float(kflat[i + 1]), float(kflat[i + 2])
            triplets.append(f"{x / img_w:.6f} {y / img_h:.6f} {int(v)}")
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} " + " ".join(triplets))
    return lines


def obb_label_lines(example: dict[str, Any], img_w: int, img_h: int) -> list[str]:
    objects = example["objects"]
    bboxes = objects["bbox"]
    cats = objects["category"]
    lines: list[str] = []
    for bbox, cat in zip(bboxes, cats):
        vals = [float(v) for v in bbox]
        cid = int(cat)
        if len(vals) == 8:
            pts = " ".join(f"{vals[i] / img_w:.6f} {vals[i + 1] / img_h:.6f}" for i in range(0, 8, 2))
            lines.append(f"{cid} {pts}")
        elif len(vals) == 5:
            cx, cy, w, h, theta = vals
            cxn, cyn = cx / img_w, cy / img_h
            wn, hn = w / img_w, h / img_h
            lines.append(f"{cid} {cxn:.6f} {cyn:.6f} {wn:.6f} {hn:.6f} {theta:.6f}")
        else:
            raise SystemExit(
                f"obb_yolo expects each bbox to have 5 (cx,cy,w,h,theta) or 8 (corner x,y...) "
                f"values; got len={len(vals)}"
            )
    return lines


def segment_label_lines(example: dict[str, Any], img_w: int, img_h: int, mask_col: str) -> list[str]:
    import cv2

    mask = example[mask_col]
    arr = np.array(mask if not hasattr(mask, "convert") else mask.convert("L"))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    lines: list[str] = []
    for cls_id in np.unique(arr):
        cid = int(cls_id)
        if cid == 0 or cid == 255:
            continue
        binary = ((arr == cls_id).astype(np.uint8)) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) < 3:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, max(0.001 * peri, 1e-4), True)
            pts = approx.squeeze(1)
            if len(pts) < 3:
                continue
            flat: list[str] = []
            for x, y in pts:
                flat.append(f"{float(x) / img_w:.6f}")
                flat.append(f"{float(y) / img_h:.6f}")
            lines.append(f"{cid} " + " ".join(flat))
    return lines


def write_yolo_det_yaml(root: Path, id2label: dict[int, str]) -> Path:
    names = {i: id2label[i] for i in sorted(id2label.keys())}
    payload = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    path = root / "data.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def collect_mask_label_values(ds: Dataset, mask_col: str, max_samples: int = 256) -> list[int]:
    found: set[int] = set()
    n = min(len(ds), max_samples)
    for i in range(n):
        m = ds[i][mask_col]
        arr = np.array(m if not hasattr(m, "convert") else m.convert("L"))
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        for v in np.unique(arr):
            vi = int(v)
            if vi not in (0, 255):
                found.add(vi)
    out = sorted(found)
    return out if out else [1]


def resolve_mask_column(ds: Dataset, explicit: Optional[str]) -> str:
    if explicit and explicit in ds.column_names:
        return explicit
    for c in ("label", "annotation", "mask", "segmentation_mask", "segmentation"):
        if c in ds.column_names:
            return c
    raise SystemExit(
        "segment_yolo: could not find a mask column; set yolo.mask_column in YAML "
        f"(columns: {ds.column_names})"
    )


def prepare_classify_dataset(
    model_args: ModelArguments,
    data_args: YoloDataArguments,
    training_args: TrainingArguments,
) -> tuple[DatasetDict, dict[int, str], str]:
    dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    label_col = data_args.label_column
    if label_col not in dataset["train"].column_names:
        label_col = "labels" if "labels" in dataset["train"].column_names else label_col

    feat = dataset["train"].features.get(label_col)
    cls_key = "_yolo_cls_id"

    if feat is not None and hasattr(feat, "names") and feat.names:
        id2label = {i: str(n) for i, n in enumerate(feat.names)}

        def _pass(example: dict[str, Any]) -> dict[str, Any]:
            example[cls_key] = int(example[label_col])
            return example

        for k in list(dataset.keys()):
            dataset[k] = dataset[k].map(_pass)
    else:
        labels_set: set[Any] = set()
        for ex in dataset["train"]:
            labels_set.add(ex[label_col])
        sorted_labels = sorted(labels_set, key=lambda x: str(x))
        id2label = {i: str(sorted_labels[i]) for i in range(len(sorted_labels))}
        rev: dict[Any, int] = {}
        for i, lab in enumerate(sorted_labels):
            rev[lab] = i
            rev[str(lab)] = i

        def _map_lab(example: dict[str, Any]) -> dict[str, Any]:
            v = example[label_col]
            rid = rev.get(v)
            if rid is None:
                rid = rev.get(str(v))
            if rid is None:
                raise ValueError(f"Unknown label value {v!r} for column {label_col!r}")
            example[cls_key] = rid
            return example

        for k in list(dataset.keys()):
            dataset[k] = dataset[k].map(_map_lab)

    dataset["train"] = dataset["train"].shuffle(seed=training_args.seed)
    tv_split = None if "validation" in dataset else data_args.train_val_split
    if isinstance(tv_split, float) and tv_split > 0.0:
        split = dataset["train"].train_test_split(tv_split, seed=training_args.seed)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    val_key = "validation" if "validation" in dataset else "test"
    if val_key not in dataset:
        raise SystemExit("classify_yolo needs a validation/test split or train_val_split.")

    if data_args.max_train_samples is not None:
        n = min(data_args.max_train_samples, len(dataset["train"]))
        dataset["train"] = dataset["train"].select(range(n))
    if data_args.max_eval_samples is not None:
        n = min(data_args.max_eval_samples, len(dataset[val_key]))
        dataset[val_key] = dataset[val_key].select(range(n))

    return dataset, id2label, cls_key


def export_classify_split(split_ds: Dataset, split_root: Path, cls_key: str) -> None:
    split_root.mkdir(parents=True, exist_ok=True)
    per_class: dict[int, int] = {}
    for idx in range(len(split_ds)):
        example = split_ds[idx]
        image = example["image"]
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        cid = int(example[cls_key])
        per_class[cid] = per_class.get(cid, 0) + 1
        stem = f"{cid}_{per_class[cid]:06d}"
        class_dir = split_root / str(cid)
        class_dir.mkdir(parents=True, exist_ok=True)
        image.save(class_dir / f"{stem}.jpg", format="JPEG", quality=95)


def write_yolo_cls_yaml(root: Path, id2label: dict[int, str]) -> Path:
    payload = {
        "path": str(root.resolve()),
        "train": "train",
        "val": "val",
        "names": {i: id2label[i] for i in sorted(id2label.keys())},
    }
    path = root / "data.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def train_loop(
    model_path: str,
    data_yaml: Path,
    ledger_task: str,
    training_args: TrainingArguments,
    data_args: YoloDataArguments,
    metrics_reader: Callable[[Path], dict[str, float]],
    start_time: float,
) -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    out_root = Path(training_args.output_dir).resolve()
    model = YOLO(model_path)
    model.train(
        data=str(data_yaml),
        epochs=int(training_args.num_train_epochs),
        imgsz=int(data_args.image_square_size),
        batch=int(training_args.per_device_train_batch_size),
        workers=int(training_args.dataloader_num_workers),
        project=str(out_root),
        name="ultralytics",
        exist_ok=True,
        seed=int(training_args.seed),
        verbose=True,
        amp=bool(training_args.fp16),
    )

    trainer = getattr(model, "trainer", None)
    save_dir = getattr(trainer, "save_dir", None) if trainer is not None else None
    run_dir = Path(save_dir).resolve() if save_dir else out_root / "ultralytics"
    train_metrics = read_train_metrics_from_csv(run_dir)
    metrics_lines = metrics_reader(run_dir)

    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    elapsed = time.time() - start_time
    emit_summary(ledger_task, metrics_lines, train_metrics, elapsed, peak_vram_mb)


def run_detect_family(
    ledger_task: str,
    model_args: ModelArguments,
    data_args: YoloDataArguments,
    training_args: TrainingArguments,
    line_builder: Callable[[dict[str, Any], int, int], list[str]],
    start_time: float,
) -> None:
    dataset, id2label = prepare_detection_like_dataset(model_args, data_args, training_args)
    if ledger_task == "pose_yolo":
        ex0 = dataset["train"][0]["objects"]
        if "keypoints" not in ex0:
            raise SystemExit(
                "pose_yolo requires HF detection samples with objects['keypoints'] "
                "(COCO-style flat [x,y,v] triplets per instance)."
            )
    yolo_root = Path(training_args.output_dir) / "yolo_dataset"
    if yolo_root.exists():
        shutil.rmtree(yolo_root)
    train_img = yolo_root / "images" / "train"
    train_lbl = yolo_root / "labels" / "train"
    val_img = yolo_root / "images" / "val"
    val_lbl = yolo_root / "labels" / "val"
    val_key = "validation" if "validation" in dataset else "test"

    def _lb(ex: dict[str, Any], iw: int, ih: int) -> list[str]:
        return line_builder(ex, iw, ih)

    export_split_detect(dataset["train"], train_img, train_lbl, "tr", _lb)
    export_split_detect(dataset[val_key], val_img, val_lbl, "va", _lb)
    data_yaml = write_yolo_det_yaml(yolo_root, id2label)
    logger.info("Wrote YOLO dataset under %s", data_yaml)
    train_loop(
        model_args.model_name_or_path,
        data_yaml,
        ledger_task,
        training_args,
        data_args,
        read_detect_metrics,
        start_time,
    )


def run_segment_yolo(
    model_args: ModelArguments,
    data_args: YoloDataArguments,
    training_args: TrainingArguments,
    start_time: float,
) -> None:
    dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    dataset["train"] = dataset["train"].shuffle(seed=training_args.seed)
    tv_split = None if "validation" in dataset else data_args.train_val_split
    if isinstance(tv_split, float) and tv_split > 0.0:
        split = dataset["train"].train_test_split(tv_split, seed=training_args.seed)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]
    val_key = "validation" if "validation" in dataset else "test"
    if val_key not in dataset:
        raise SystemExit("segment_yolo needs val/test or train_val_split.")

    if data_args.max_train_samples is not None:
        n = min(data_args.max_train_samples, len(dataset["train"]))
        dataset["train"] = dataset["train"].select(range(n))
    if data_args.max_eval_samples is not None:
        n = min(data_args.max_eval_samples, len(dataset[val_key]))
        dataset[val_key] = dataset[val_key].select(range(n))

    mask_col = resolve_mask_column(dataset["train"], data_args.mask_column)
    tr_c = collect_mask_label_values(dataset["train"], mask_col)
    va_c = collect_mask_label_values(dataset[val_key], mask_col)
    classes = sorted(set(tr_c) | set(va_c))
    if not classes:
        classes = [1]
    id2label = {i: f"class_{c}" for i, c in enumerate(classes)}

    yolo_root = Path(training_args.output_dir) / "yolo_dataset"
    if yolo_root.exists():
        shutil.rmtree(yolo_root)

    def seg_builder(ex: dict[str, Any], iw: int, ih: int) -> list[str]:
        return segment_label_lines(ex, iw, ih, mask_col)

    export_split_detect(dataset["train"], yolo_root / "images" / "train", yolo_root / "labels" / "train", "tr", seg_builder)
    export_split_detect(dataset[val_key], yolo_root / "images" / "val", yolo_root / "labels" / "val", "va", seg_builder)
    remap = {old: i for i, old in enumerate(classes)}
    data_yaml = write_yolo_det_yaml(yolo_root, {i: id2label[i] for i in range(len(classes))})

    def _rewrite_labels(dir_path: Path) -> None:
        for p in dir_path.glob("*.txt"):
            out_lines = []
            for line in p.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if not parts:
                    continue
                old_cls = int(float(parts[0]))
                if old_cls not in remap:
                    continue
                parts[0] = str(remap[old_cls])
                out_lines.append(" ".join(parts))
            p.write_text("\n".join(out_lines), encoding="utf-8")

    _rewrite_labels(yolo_root / "labels" / "train")
    _rewrite_labels(yolo_root / "labels" / "val")

    logger.info("Wrote YOLO-seg dataset under %s (mask column=%s)", data_yaml, mask_col)
    train_loop(
        model_args.model_name_or_path,
        data_yaml,
        "segment_yolo",
        training_args,
        data_args,
        read_segment_metrics,
        start_time,
    )


def run_classify_yolo(
    model_args: ModelArguments,
    data_args: YoloDataArguments,
    training_args: TrainingArguments,
    start_time: float,
) -> None:
    dataset, id2label, cls_key = prepare_classify_dataset(model_args, data_args, training_args)
    val_key = "validation" if "validation" in dataset else "test"
    yolo_root = Path(training_args.output_dir) / "yolo_cls_dataset"
    if yolo_root.exists():
        shutil.rmtree(yolo_root)
    export_classify_split(dataset["train"], yolo_root / "train", cls_key)
    export_classify_split(dataset[val_key], yolo_root / "val", cls_key)
    data_yaml = write_yolo_cls_yaml(yolo_root, id2label)
    logger.info("Wrote YOLO-cls dataset under %s", data_yaml)
    train_loop(
        model_args.model_name_or_path,
        data_yaml,
        "classify_yolo",
        training_args,
        data_args,
        read_classify_metrics,
        start_time,
    )


def main() -> None:
    start_time = time.time()

    if len(sys.argv) != 2 or not sys.argv[1].endswith((".yaml", ".yml", ".json")):
        raise SystemExit("Usage: train_ultralytics.py <config.yaml|config.json>")

    cfg_path = Path(sys.argv[1]).resolve()
    raw_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw_cfg, dict):
        raise SystemExit("Config must be a YAML mapping")
    ledger_task = raw_cfg.get("task_type")
    if ledger_task not in LEDGER_TASKS:
        raise SystemExit(
            f"task_type must be one of {sorted(LEDGER_TASKS)}; got {ledger_task!r}"
        )

    parser = HfArgumentParser((ModelArguments, YoloDataArguments, TrainingArguments))
    if cfg_path.suffix.lower() in {".yaml", ".yml"}:
        model_args, data_args, training_args = parser.parse_yaml_file(
            yaml_file=str(cfg_path), allow_extra_keys=True
        )
    else:
        model_args, data_args, training_args = parser.parse_json_file(json_file=str(cfg_path))

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("hfjob")
    if hf_token:
        login(token=hf_token)
        training_args.hub_token = hf_token
    elif training_args.push_to_hub:
        logger.warning("HF_TOKEN not found; Hub push may fail.")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
    trackio.init(project=training_args.output_dir, name=training_args.run_name)

    if ledger_task in ("detect_yolo", "track_yolo"):
        run_detect_family(ledger_task, model_args, data_args, training_args, detect_label_lines, start_time)
    elif ledger_task == "pose_yolo":
        run_detect_family(ledger_task, model_args, data_args, training_args, pose_label_lines, start_time)
    elif ledger_task == "obb_yolo":
        run_detect_family(ledger_task, model_args, data_args, training_args, obb_label_lines, start_time)
    elif ledger_task == "segment_yolo":
        run_segment_yolo(model_args, data_args, training_args, start_time)
    elif ledger_task == "classify_yolo":
        run_classify_yolo(model_args, data_args, training_args, start_time)
    else:
        raise SystemExit(f"Unhandled task_type: {ledger_task}")


if __name__ == "__main__":
    main()
