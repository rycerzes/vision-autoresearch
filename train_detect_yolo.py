"""Fine-tune Ultralytics YOLO on Hugging Face Hub detection datasets.

This is stable infrastructure - do NOT edit during experiments.
Experiments modify config YAMLs only.
"""

from __future__ import annotations

import csv
import logging
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import login
from transformers import HfArgumentParser, TrainingArguments

import trackio
from train_detect import ModelArguments, detect_bbox_format_from_samples, sanitize_dataset
from ultralytics import YOLO

logger = logging.getLogger(__name__)


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


def emit_summary(
    metrics: dict[str, float],
    train_metrics: dict[str, Any],
    training_seconds: float,
    peak_vram_mb: float,
) -> None:
    print("\n--- VISION AUTORESEARCH SUMMARY ---")
    print("task_type: detect_yolo")
    print(f"mAP: {metrics.get('map', 0.0)}")
    print(f"mAP_50: {metrics.get('map_50', 0.0)}")
    print(f"mAR: {metrics.get('mar', 0.0)}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"peak_vram_mb: {peak_vram_mb:.0f}")
    print(f"train_loss: {train_metrics.get('train_loss', 0.0)}")
    print(f"num_train_epochs: {train_metrics.get('epoch', 0)}")
    print("--- END SUMMARY ---")


def discover_categories_and_remap(dataset: DatasetDict) -> dict[int, str]:
    """Discover class names and remap string categories to integers (mutates dataset)."""
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
        logger.info("Category feature is not ClassLabel — scanning dataset to discover labels...")
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
        logger.info("Remapping string categories to integer IDs")

        def _remap(example: dict[str, Any]) -> dict[str, Any]:
            objects = example["objects"]
            objects["category"] = [label2id[c] for c in objects["category"]]
            example["objects"] = objects
            return example

        for split_name in list(dataset.keys()):
            dataset[split_name] = dataset[split_name].map(_remap)

    return id2label


def bbox_to_yolo_line(bbox: list[float], cat_id: int, img_w: int, img_h: int) -> str | None:
    """COCO xywh (pixels) → YOLO normalized cx cy w h."""
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


def export_split_to_yolo(split_ds: Dataset, images_dir: Path, labels_dir: Path, prefix: str) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(len(split_ds)):
        example = split_ds[idx]
        image = example["image"]
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        img_w, img_h = image.size
        stem = f"{prefix}_{idx:06d}"
        img_path = images_dir / f"{stem}.jpg"
        image.save(img_path, format="JPEG", quality=95)

        objects = example["objects"]
        bboxes = objects["bbox"]
        cats = objects["category"]
        lines: list[str] = []
        for bbox, cat in zip(bboxes, cats):
            cid = int(cat)
            line = bbox_to_yolo_line(list(bbox), cid, img_w, img_h)
            if line:
                lines.append(line)
        (labels_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")


def write_yolo_data_yaml(root: Path, id2label: dict[int, str]) -> Path:
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


def read_box_metrics_from_results_csv(run_dir: Path) -> dict[str, float]:
    """Read validation box metrics from Ultralytics results.csv (last epoch)."""
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return {"map": 0.0, "map_50": 0.0, "mar": 0.0}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"map": 0.0, "map_50": 0.0, "mar": 0.0}
    last = rows[-1]

    def pick(keys: tuple[str, ...]) -> float:
        for k in keys:
            raw = last.get(k)
            if raw in (None, ""):
                continue
            try:
                return float(raw)
            except (TypeError, ValueError):
                continue
        return 0.0

    return {
        "map": pick(("metrics/mAP50-95(B)", "metrics/mAP50-95(M)")),
        "map_50": pick(("metrics/mAP50(B)", "metrics/mAP50(M)")),
        "mar": pick(("metrics/recall(B)", "metrics/recall(M)")),
    }


def prepare_detection_dataset(
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
        logger.info("Converting bboxes from xyxy (Pascal VOC) → xywh before YOLO export")
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


def main() -> None:
    start_time = time.time()

    parser = HfArgumentParser((ModelArguments, YoloDataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".yml")):
        model_args, data_args, training_args = parser.parse_yaml_file(
            yaml_file=os.path.abspath(sys.argv[1]), allow_extra_keys=True
        )
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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

    dataset, id2label = prepare_detection_dataset(model_args, data_args, training_args)

    yolo_root = Path(training_args.output_dir) / "yolo_dataset"
    if yolo_root.exists():
        shutil.rmtree(yolo_root)

    train_img = yolo_root / "images" / "train"
    train_lbl = yolo_root / "labels" / "train"
    val_img = yolo_root / "images" / "val"
    val_lbl = yolo_root / "labels" / "val"

    val_key = "validation" if "validation" in dataset else "test"
    export_split_to_yolo(dataset["train"], train_img, train_lbl, "tr")
    export_split_to_yolo(dataset[val_key], val_img, val_lbl, "va")

    data_yaml = write_yolo_data_yaml(yolo_root, id2label)
    logger.info("Wrote YOLO dataset under %s", data_yaml)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model = YOLO(model_args.model_name_or_path)
    model.train(
        data=str(data_yaml),
        epochs=int(training_args.num_train_epochs),
        imgsz=int(data_args.image_square_size),
        batch=int(training_args.per_device_train_batch_size),
        workers=int(training_args.dataloader_num_workers),
        project=str(training_args.output_dir),
        name="ultralytics",
        exist_ok=True,
        seed=int(training_args.seed),
        verbose=True,
        amp=bool(training_args.fp16),
    )

    run_dir = Path(training_args.output_dir) / "ultralytics"
    train_metrics = read_train_metrics_from_csv(run_dir)
    metrics = read_box_metrics_from_results_csv(run_dir)

    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    elapsed = time.time() - start_time
    emit_summary(metrics, train_metrics, elapsed, peak_vram_mb)


if __name__ == "__main__":
    main()
