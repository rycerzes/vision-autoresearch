"""Ultralytics YOLO training for multiple CV tasks using Hugging Face Hub datasets.

Stable infrastructure — the trainer entry is a validated ``RunContract`` file only:

  ``train_ultralytics.py <run-contract.yaml|run-contract.json>``

Supported ``task_type`` values (default promotion primary in parentheses; see
``vision_lab.task_registry`` and ``vision_lab.metrics``):

  detect_yolo (mAP), track_yolo (mAP), segment_yolo (mask_map mask branch),
  classify_yolo (accuracy), pose_yolo (mAP), obb_yolo (mAP).

Summary lines use canonical metric keys from ``vision_lab.metrics.METRICS`` where
applicable; remaining numeric keys are emitted after those in sorted order.

``track_yolo`` trains a detector (Ultralytics has no ``train`` mode for tracking);
use the same detection dataset contract; tracking inference is separate.

Ultralytics hyperparameters
  Contract field ``training.hyperparameters`` may include ``ultralytics_train`` (mapping)
  passed to the model's ``.train()`` method, merged with Ultralytics defaults. Keys not
  listed there are filled from ``TrainingArguments`` / ``YoloDataArguments`` where they overlap
  Ultralytics names: ``epochs`` ← ``num_train_epochs``,
  ``batch`` ← ``per_device_train_batch_size``, ``imgsz`` ← ``image_square_size``,
  ``workers`` ← ``dataloader_num_workers``, ``seed``, ``amp`` ← ``fp16``.
  HF-only fields do **not** affect training unless mirrored under ``ultralytics_train``
  (e.g. ``lr0``, ``weight_decay``, ``warmup_epochs``, ``cos_lr``).
  Script-owned keys ``data``, ``project``, ``name``, and ``exist_ok`` are always set
  (values under ``ultralytics_train`` for those keys are ignored).

  Optional string key ``ultralytics_train.trainer`` names a trainer class (e.g.
  ``YOLOEPESegTrainer``, ``WorldTrainer``); see Ultralytics docs for YOLOE / YOLO-World.

Ultralytics bridge (optional mapping under contract ``training.hyperparameters`` as ``ultralytics_bridge``)
  ``model_class`` (``auto`` | ``yolo`` | ``yoloe`` | ``yolo_world`` | ``rtdetr``):
    Which Ultralytics entry type loads ``model_name_or_path``. ``auto`` uses
    ``YOLO(...)``, which switches to YOLOWorld / YOLOE / RT-DETR when the weight
    stem matches Ultralytics conventions.
  ``yoloe_training`` (``auto`` | ``full`` | ``linear_probe``): when training YOLOE
    and ``trainer`` is unset, selects ``YOLOE*Trainer`` vs ``YOLOEPE*Trainer``
    (PE = linear probing in Ultralytics). ``auto`` defaults to ``full``.
  ``yoloe_pretrained_weights``: optional path passed as ``pretrained=`` to
    ``.train()`` (required for some YOLO-from-YAML + seg-weights flows; see
    https://docs.ultralytics.com/models/yoloe/ ).
  ``sync_class_names`` (default ``true``): for YOLO-World / YOLOE, call
    ``set_classes`` with names from the exported dataset.

YOLO-NAS: Ultralytics documents **no training** for NAS weights (val / predict /
export only). This script aborts with a clear error if a NAS checkpoint is used
for a training task. See https://docs.ultralytics.com/models/yolo-nas/

References: https://docs.ultralytics.com/modes/train/
  https://docs.ultralytics.com/models/yoloe/
  https://docs.ultralytics.com/models/yolo-world/
"""

from __future__ import annotations

import csv
import inspect
import logging
import math
import os
import re
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import trackio
import yaml
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import login
from transformers import HfArgumentParser, TrainingArguments
from ultralytics.models.rtdetr.model import RTDETR
from ultralytics.models.yolo.model import YOLO, YOLOE, YOLOWorld
from ultralytics.utils import DEFAULT_CFG

_SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from vision_lab.hf_vision.detect_train import (
    ModelArguments,
    detect_bbox_format_from_samples,
    sanitize_dataset,
)
from vision_lab.metrics import METRICS

logger = logging.getLogger(__name__)


def _require_training_output_dir(training_args: TrainingArguments) -> Path:
    od = training_args.output_dir
    if od is None or not str(od).strip():
        raise SystemExit("output_dir must be set in config for Ultralytics training/export paths.")
    return Path(od).expanduser().resolve()


# Keys Ultralytics documents for training; used only to warn on likely typos.
_KNOWN_ULTRATRAIN_KEYS = frozenset(vars(DEFAULT_CFG).keys())

# Always set by this bridge (HF ``output_dir`` / exported dataset).
_RESERVED_ULTRATRAIN_KEYS = frozenset({"data", "project", "name", "exist_ok"})

# Optional YAML: ultralytics_bridge (see module docstring).
_TRAINER_REGISTRY: dict[str, type[Any]] | None = None


def _coerce_ultralytics_bridge(raw: Any, config_path: Path) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SystemExit(
            f"{config_path}: ultralytics_bridge must be a mapping, got {type(raw).__name__}"
        )
    return {str(k): v for k, v in raw.items()}


def _stem_lower(model_path: str) -> str:
    return Path(str(model_path)).stem.lower()


def _looks_like_yolo_nas_weights(model_path: str) -> bool:
    s = _stem_lower(model_path)
    return "yolo_nas" in s or s.startswith("yolonas")


def _trainer_name_registry() -> dict[str, type[Any]]:
    """Map YAML ``trainer`` string → class. Lazily populated for optional Ultralytics extras."""
    global _TRAINER_REGISTRY
    if _TRAINER_REGISTRY is not None:
        return _TRAINER_REGISTRY
    reg: dict[str, type[Any]] = {}
    try:
        from ultralytics.models.yolo.world.train import WorldTrainer

        reg["WorldTrainer"] = WorldTrainer
    except ImportError:
        pass
    try:
        from ultralytics.models.yolo import yoloe as _yoloe_mod

        for _name in (
            "YOLOEPETrainer",
            "YOLOEPESegTrainer",
            "YOLOETrainer",
            "YOLOESegTrainer",
            "YOLOETrainerFromScratch",
            "YOLOESegTrainerFromScratch",
            "YOLOEPEFreeTrainer",
            "YOLOEVPTrainer",
            "YOLOESegVPTrainer",
        ):
            if hasattr(_yoloe_mod, _name):
                reg[_name] = getattr(_yoloe_mod, _name)
    except ImportError:
        pass
    _TRAINER_REGISTRY = reg
    return reg


def _resolve_trainer_class(name: str) -> type[Any]:
    reg = _trainer_name_registry()
    key = str(name).strip()
    if key in reg:
        return reg[key]
    raise SystemExit(
        f"Unknown ultralytics_train.trainer / ultralytics_bridge.trainer {key!r}. "
        f"Known names: {sorted(reg)}"
    )


def _yoloe_task_for_ledger(ledger_task: str) -> str:
    if ledger_task == "segment_yolo":
        return "segment"
    return "detect"


def load_ultralytics_model(model_path: str, ledger_task: str, bridge: dict[str, Any]) -> Any:
    """Construct the Ultralytics model object per ``ultralytics_bridge.model_class``."""
    if _looks_like_yolo_nas_weights(model_path):
        raise SystemExit(
            "YOLO-NAS checkpoints cannot be trained in Ultralytics (inference / val / export only). "
            "Use a standard YOLO or YOLOE weight for training, or run val/predict outside this benchmark. "
            "See https://docs.ultralytics.com/models/yolo-nas/"
        )
    raw_mc = bridge.get("model_class", "auto")
    mc = str(raw_mc).strip().lower() if raw_mc is not None else "auto"
    if mc in ("auto", "", "yolo"):
        return YOLO(model_path)
    if mc == "yoloe":
        return YOLOE(model_path, task=_yoloe_task_for_ledger(ledger_task))
    if mc in ("yolo_world", "yoloworld", "world"):
        return YOLOWorld(model_path)
    if mc in ("rtdetr", "rt-detr", "rt_detr"):
        return RTDETR(model_path)
    raise SystemExit(
        f"ultralytics_bridge.model_class must be one of auto, yolo, yoloe, yolo_world, rtdetr; got {raw_mc!r}"
    )


def _maybe_sync_open_vocab_class_names(
    model: Any, ordered_names: list[str], bridge: dict[str, Any]
) -> None:
    if not ordered_names:
        return
    sync = bridge.get("sync_class_names", True)
    if sync is False or str(sync).lower() in {"0", "false", "no"}:
        return
    if isinstance(model, YOLOWorld):
        model.set_classes(ordered_names)
        logger.info("YOLO-World: set_classes(%d names) from dataset", len(ordered_names))
        return
    if isinstance(model, YOLOE):
        try:
            model.set_classes(ordered_names)
            logger.info("YOLOE: set_classes(%d names) from dataset", len(ordered_names))
        except Exception as exc:
            logger.warning("YOLOE set_classes skipped: %s", exc)


def _resolve_trainer_for_yoloe(ledger_task: str, mode: str) -> type[Any] | None:
    """Pick default YOLOE trainer when ``yoloe_training`` is full vs linear_probe (Ultralytics PE trainers)."""
    mode = (mode or "auto").strip().lower()
    if mode == "auto":
        mode = "full"
    reg = _trainer_name_registry()
    if ledger_task == "segment_yolo":
        if mode == "linear_probe":
            return reg.get("YOLOEPESegTrainer")
        if mode == "full":
            return reg.get("YOLOESegTrainer")
        raise SystemExit(f"yoloe_training must be auto, full, or linear_probe; got {mode!r}")
    if ledger_task in ("detect_yolo", "track_yolo", "pose_yolo", "obb_yolo"):
        if mode == "linear_probe":
            return reg.get("YOLOEPETrainer")
        if mode == "full":
            return reg.get("YOLOETrainer")
        raise SystemExit(f"yoloe_training must be auto, full, or linear_probe; got {mode!r}")
    return None


def _pick_trainer_class(
    model: Any,
    ledger_task: str,
    bridge: dict[str, Any],
    explicit_trainer: type[Any] | None,
    bridge_trainer_name: str | None,
) -> type[Any] | None:
    if explicit_trainer is not None:
        return explicit_trainer
    if bridge_trainer_name:
        return _resolve_trainer_class(bridge_trainer_name)
    if isinstance(model, YOLOE):
        y_mode = bridge.get("yoloe_training", "auto")
        return _resolve_trainer_for_yoloe(ledger_task, str(y_mode))
    return None


def _coerce_ultralytics_train_block(raw: Any, config_path: Path) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SystemExit(
            f"{config_path}: ultralytics_train must be a mapping, got {type(raw).__name__}"
        )
    return {str(k): v for k, v in raw.items()}


def _warn_unknown_ultralytics_keys(user_keys: set[str]) -> None:
    unknown = sorted(user_keys - _KNOWN_ULTRATRAIN_KEYS)
    if unknown and logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "ultralytics_train keys not in Ultralytics DEFAULT_CFG (may still be valid): %s",
            unknown,
        )


def build_ultralytics_train_kwargs(
    *,
    data_yaml: Path,
    out_root: Path,
    training_args: TrainingArguments,
    data_args: YoloDataArguments,
    ultralytics_train: dict[str, Any],
    bridge: dict[str, Any],
) -> tuple[dict[str, Any], type[Any] | None]:
    """Build kwargs for ``model.train()``; returns ``(train_kwargs, explicit_trainer_class)``."""
    merged: dict[str, Any] = dict(ultralytics_train)
    trainer_spec = merged.pop("trainer", None)
    explicit_trainer: type[Any] | None = None
    if isinstance(trainer_spec, str) and trainer_spec.strip():
        explicit_trainer = _resolve_trainer_class(trainer_spec.strip())
    elif inspect.isclass(trainer_spec):
        explicit_trainer = trainer_spec

    collisions = _RESERVED_ULTRATRAIN_KEYS & merged.keys()
    if collisions:
        logger.info("ultralytics_train: ignoring script-owned keys %s", sorted(collisions))
    for k in _RESERVED_ULTRATRAIN_KEYS:
        merged.pop(k, None)

    _warn_unknown_ultralytics_keys(set(merged.keys()))

    yp = bridge.get("yoloe_pretrained_weights")
    if yp is not None and str(yp).strip():
        merged["pretrained"] = str(yp)

    defaults_from_hf = {
        "epochs": int(training_args.num_train_epochs),
        "batch": int(training_args.per_device_train_batch_size),
        "imgsz": int(data_args.image_square_size),
        "workers": int(training_args.dataloader_num_workers),
        "seed": int(training_args.seed),
        "amp": bool(training_args.fp16),
    }
    for key, value in defaults_from_hf.items():
        if key not in merged:
            merged[key] = value

    merged["data"] = str(data_yaml)
    merged["project"] = str(out_root)
    merged["name"] = "ultralytics"
    merged["exist_ok"] = True
    merged.setdefault("verbose", True)

    return merged, explicit_trainer


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
    dataset_config_name: str | None = field(default=None, metadata={"help": "Dataset config name."})
    dataset_revision: str | None = field(
        default=None,
        metadata={"help": "Optional Hub dataset revision (commit, tag, or branch)."},
    )
    train_val_split: float | None = field(
        default=0.15,
        metadata={"help": "Holdout fraction from train when no validation split exists."},
    )
    image_square_size: int = field(default=640, metadata={"help": "YOLO imgsz."})
    max_train_samples: int | None = field(default=None, metadata={"help": "Cap training samples."})
    max_eval_samples: int | None = field(default=None, metadata={"help": "Cap validation samples."})
    label_column: str = field(
        default="label",
        metadata={"help": "Classification label column (classify_yolo)."},
    )
    mask_column: str | None = field(
        default=None,
        metadata={"help": "Semantic mask column for segment_yolo (auto if unset)."},
    )
    objects_category_field: str | None = field(
        default=None,
        metadata={
            "help": (
                "Which ``objects`` sub-field holds per-instance class ids or names for detection-like tasks. "
                "Use ``auto`` (default) to pick the first present among category, label, categories "
                "(same order as prepare.py). Set to ``label`` for datasets that use objects['label'] only."
            )
        },
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
    """Mask-branch detection metrics from Ultralytics (honest mAP names; not IoU)."""
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return {"mask_map": 0.0}
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    if not rows:
        return {"mask_map": 0.0}
    last = rows[-1]
    map50_m = pick_csv_metric(last, ("metrics/mAP50(M)",))
    return {"mask_map": map50_m}


def _assert_pose_keypoints(dataset: Any, ledger_task: str) -> None:
    if ledger_task != "pose_yolo":
        return
    objects = dataset["train"][0].get("objects")
    if not isinstance(objects, dict):
        raise SystemExit("pose_yolo requires each example to have an objects mapping.")
    if objects.get("keypoints") is None:
        raise SystemExit(
            "pose_yolo requires objects['keypoints'] per instance (same length as bbox); "
            "synthetic keypoints are not supported."
        )


def emit_summary(
    ledger_task: str,
    metrics_lines: dict[str, float],
    train_metrics: dict[str, Any],
    training_seconds: float,
    peak_vram_mb: float,
) -> None:
    print("\n--- VISION AUTORESEARCH SUMMARY ---")
    print(f"task_type: {ledger_task}")
    emitted: set[str] = set()
    for key in METRICS:
        if key in metrics_lines:
            print(f"{key}: {metrics_lines[key]}")
            emitted.add(key)
    for key, val in sorted(metrics_lines.items()):
        if key not in emitted:
            print(f"{key}: {val}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"peak_vram_mb: {peak_vram_mb:.0f}")
    print(f"train_loss: {train_metrics.get('train_loss', 0.0)}")
    print(f"num_train_epochs: {train_metrics.get('epoch', 0)}")
    print("--- END SUMMARY ---")


_OBJECTS_CATEGORY_KEYS = ("category", "label", "categories")


def _objects_category_source_key(sample_objects: dict[str, Any], explicit: str | None) -> str:
    """Resolve which objects sub-field holds class labels (aligned with prepare.py detection inspection)."""
    raw = str(explicit).strip() if explicit is not None else ""
    raw_lower = raw.lower()
    if raw_lower and raw_lower != "auto":
        if raw in sample_objects:
            return raw
        for smk in sample_objects:
            if smk.lower() == raw_lower:
                return smk
        raise SystemExit(
            f"objects_category_field={explicit!r} not found under objects; "
            f"keys present: {sorted(sample_objects)}"
        )
    for k in _OBJECTS_CATEGORY_KEYS:
        if k in sample_objects:
            return k
    raise SystemExit(
        "Detection-like YOLO export requires objects to contain one of: "
        "category, label, or categories (see prepare.py). "
        f"Found objects keys: {sorted(sample_objects)}"
    )


def _category_classlabel_names_from_features(ds_train: Dataset) -> list[str] | None:
    """Read ClassLabel.names from objects.{category|label|categories} if present."""
    try:
        objects_feat = ds_train.features["objects"]
    except KeyError:
        return None
    for key in _OBJECTS_CATEGORY_KEYS:
        try:
            if isinstance(objects_feat, dict):
                sub = objects_feat[key]
            elif hasattr(objects_feat, "feature"):
                sub = objects_feat.feature[key]
            else:
                continue
            cat_feature = sub.feature if hasattr(sub, "feature") else sub
            if hasattr(cat_feature, "names") and cat_feature.names:
                return list(cat_feature.names)
        except (KeyError, AttributeError, TypeError):
            continue
    return None


def ensure_objects_have_category_column(dataset: DatasetDict, explicit_field: str | None) -> None:
    """Copy ``label`` / ``categories`` into ``category`` when the canonical key is absent."""
    sample_o = dataset["train"][0]["objects"]
    if not isinstance(sample_o, dict):
        raise SystemExit(
            "Expected objects to be a dict with bbox + category/label; "
            "got a non-dict objects column."
        )
    src = _objects_category_source_key(sample_o, explicit_field)
    if src == "category":
        return

    def _copy_into_category(example: dict[str, Any]) -> dict[str, Any]:
        o = dict(example["objects"])
        if "category" not in o:
            o["category"] = o[src]
        example["objects"] = o
        return example

    for split_name in list(dataset.keys()):
        dataset[split_name] = dataset[split_name].map(_copy_into_category)


def discover_categories_and_remap(dataset: DatasetDict) -> dict[int, str]:
    categories = _category_classlabel_names_from_features(dataset["train"])

    if categories is None:
        logger.info("Category feature is not ClassLabel — scanning dataset...")
        unique_cats: set[Any] = set()
        for raw_example in dataset["train"]:
            example = cast(dict[str, Any], raw_example)
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


def _load_yolo_hub_dataset(
    model_args: ModelArguments,
    data_args: YoloDataArguments,
) -> DatasetDict:
    kw: dict[str, Any] = {
        "cache_dir": model_args.cache_dir,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if data_args.dataset_revision:
        kw["revision"] = data_args.dataset_revision
    return cast(DatasetDict, load_dataset(data_args.dataset_name, data_args.dataset_config_name, **kw))


def prepare_detection_like_dataset(
    model_args: ModelArguments,
    data_args: YoloDataArguments,
    training_args: TrainingArguments,
) -> tuple[DatasetDict, dict[int, str]]:
    dataset = _load_yolo_hub_dataset(model_args, data_args)
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

    ensure_objects_have_category_column(dataset, data_args.objects_category_field)

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
    kpts_all = objects.get("keypoints")
    if kpts_all is None:
        raise SystemExit(
            "pose_yolo export requires objects['keypoints'] for each row (aligned with bbox)."
        )
    if len(kpts_all) != len(bboxes):
        raise SystemExit(
            f"pose_yolo: len(keypoints)={len(kpts_all)} does not match len(bbox)={len(bboxes)}."
        )

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
            pts = " ".join(
                f"{vals[i] / img_w:.6f} {vals[i + 1] / img_h:.6f}" for i in range(0, 8, 2)
            )
            lines.append(f"{cid} {pts}")
        elif len(vals) == 5:
            cx, cy, w, h, theta = vals
            cxn, cyn = cx / img_w, cy / img_h
            wn, hn = w / img_w, h / img_h
            lines.append(f"{cid} {cxn:.6f} {cyn:.6f} {wn:.6f} {hn:.6f} {theta:.6f}")
        elif len(vals) == 4:
            raise SystemExit(
                "obb_yolo does not accept 4-value xywh boxes; use 5 values (cx,cy,w,h,theta) or "
                "8 corner coordinates. For axis-aligned xywh detection use task detect_yolo."
            )
        else:
            raise SystemExit(
                f"obb_yolo expects each bbox to have 5 (cx,cy,w,h,theta) or 8 (corner x,y ...) "
                f"values; got len={len(vals)}"
            )
    return lines


def segment_label_lines(
    example: dict[str, Any], img_w: int, img_h: int, mask_col: str
) -> list[str]:
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


def write_yolo_det_yaml(
    root: Path,
    id2label: dict[int, str],
    kpt_shape: list[int] | None = None,
) -> Path:
    names = {i: id2label[i] for i in sorted(id2label.keys())}
    payload = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    if kpt_shape is not None:
        payload["kpt_shape"] = [int(kpt_shape[0]), int(kpt_shape[1])]
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


def resolve_mask_column(ds: Dataset, explicit: str | None) -> str:
    if not explicit or not str(explicit).strip():
        raise SystemExit(
            "segment_yolo: contract must set dataset.column_mapping role 'mask' (via compile); "
            "cannot infer mask column at runtime."
        )
    name = str(explicit).strip()
    if name not in ds.column_names:
        raise SystemExit(
            f"segment_yolo: mask column {name!r} not in dataset columns {ds.column_names}"
        )
    return name


def prepare_classify_dataset(
    model_args: ModelArguments,
    data_args: YoloDataArguments,
    training_args: TrainingArguments,
) -> tuple[DatasetDict, dict[int, str], str]:
    dataset = _load_yolo_hub_dataset(model_args, data_args)
    label_col = data_args.label_column
    if label_col not in dataset["train"].column_names:
        raise SystemExit(
            f"classify_yolo: label column {label_col!r} not in training split columns "
            f"{dataset['train'].column_names}"
        )
    cls_key = "_yolo_cls_id"
    feat = dataset["train"].features.get(label_col)

    if feat is not None and hasattr(feat, "names") and feat.names:
        id2label = {i: str(n) for i, n in enumerate(feat.names)}

        def _pass(example: dict[str, Any]) -> dict[str, Any]:
            example[cls_key] = int(example[label_col])
            return example

        for k in list(dataset.keys()):
            dataset[k] = dataset[k].map(_pass)
    else:
        labels_set: set[Any] = set()
        for raw_ex in dataset["train"]:
            ex = cast(dict[str, Any], raw_ex)
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


def ordered_class_names_from_id2label(id2label: dict[int, str]) -> list[str]:
    """Stable class-name list aligned with YOLO class indices."""
    return [id2label[i] for i in sorted(id2label.keys())]


def train_loop(
    model_path: str,
    data_yaml: Path,
    ledger_task: str,
    training_args: TrainingArguments,
    data_args: YoloDataArguments,
    metrics_reader: Callable[[Path], dict[str, float]],
    start_time: float,
    ultralytics_train: dict[str, Any],
    bridge: dict[str, Any],
    ordered_class_names: list[str] | None = None,
) -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    out_root = _require_training_output_dir(training_args)
    train_kwargs, explicit_trainer = build_ultralytics_train_kwargs(
        data_yaml=data_yaml,
        out_root=out_root,
        training_args=training_args,
        data_args=data_args,
        ultralytics_train=ultralytics_train,
        bridge=bridge,
    )
    model = load_ultralytics_model(model_path, ledger_task, bridge)
    if ordered_class_names:
        _maybe_sync_open_vocab_class_names(model, ordered_class_names, bridge)

    bridge_trainer_name = bridge.get("trainer")
    bridge_trainer_str = str(bridge_trainer_name).strip() if bridge_trainer_name else None
    trainer_cls = _pick_trainer_class(
        model,
        ledger_task,
        bridge,
        explicit_trainer,
        bridge_trainer_str,
    )
    if trainer_cls is not None:
        logger.info("Using Ultralytics trainer %s", trainer_cls.__name__)

    if trainer_cls is not None:
        model.train(trainer=trainer_cls, **train_kwargs)
    else:
        model.train(**train_kwargs)

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
    ultralytics_train: dict[str, Any],
    bridge: dict[str, Any],
) -> None:
    dataset, id2label = prepare_detection_like_dataset(model_args, data_args, training_args)
    _assert_pose_keypoints(dataset, ledger_task)
    base_out = _require_training_output_dir(training_args)
    yolo_root = base_out / "yolo_dataset"
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
    kpt_shape = [1, 3] if ledger_task == "pose_yolo" else None
    data_yaml = write_yolo_det_yaml(yolo_root, id2label, kpt_shape=kpt_shape)
    logger.info("Wrote YOLO dataset under %s", data_yaml)
    train_loop(
        model_args.model_name_or_path,
        data_yaml,
        ledger_task,
        training_args,
        data_args,
        read_detect_metrics,
        start_time,
        ultralytics_train,
        bridge,
        ordered_class_names=ordered_class_names_from_id2label(id2label),
    )


def run_segment_yolo(
    model_args: ModelArguments,
    data_args: YoloDataArguments,
    training_args: TrainingArguments,
    start_time: float,
    ultralytics_train: dict[str, Any],
    bridge: dict[str, Any],
) -> None:
    dataset = _load_yolo_hub_dataset(model_args, data_args)
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

    base_out = _require_training_output_dir(training_args)
    yolo_root = base_out / "yolo_dataset"
    if yolo_root.exists():
        shutil.rmtree(yolo_root)

    def seg_builder(ex: dict[str, Any], iw: int, ih: int) -> list[str]:
        return segment_label_lines(ex, iw, ih, mask_col)

    export_split_detect(
        dataset["train"],
        yolo_root / "images" / "train",
        yolo_root / "labels" / "train",
        "tr",
        seg_builder,
    )
    export_split_detect(
        dataset[val_key],
        yolo_root / "images" / "val",
        yolo_root / "labels" / "val",
        "va",
        seg_builder,
    )
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
        ultralytics_train,
        bridge,
        ordered_class_names=ordered_class_names_from_id2label(id2label),
    )


def run_classify_yolo(
    model_args: ModelArguments,
    data_args: YoloDataArguments,
    training_args: TrainingArguments,
    start_time: float,
    ultralytics_train: dict[str, Any],
    bridge: dict[str, Any],
) -> None:
    dataset, id2label, cls_key = prepare_classify_dataset(model_args, data_args, training_args)
    val_key = "validation" if "validation" in dataset else "test"
    base_out = _require_training_output_dir(training_args)
    yolo_root = base_out / "yolo_cls_dataset"
    if yolo_root.exists():
        shutil.rmtree(yolo_root)
    export_classify_split(dataset["train"], yolo_root / "train", cls_key)
    export_classify_split(dataset[val_key], yolo_root / "val", cls_key)
    data_yaml = write_yolo_cls_yaml(yolo_root, id2label)
    logger.info("Wrote YOLO-cls dataset under %s", data_yaml)
    # Ultralytics classify expects `data` to be a directory root, not a YAML file path.
    train_loop(
        model_args.model_name_or_path,
        yolo_root,
        "classify_yolo",
        training_args,
        data_args,
        read_classify_metrics,
        start_time,
        ultralytics_train,
        bridge,
        ordered_class_names=ordered_class_names_from_id2label(id2label),
    )


def _run_ultralytics_from_contract(contract_path: Path, *, start_time: float) -> None:
    from vision_lab.contracts.loader import load_run_contract

    contract = load_run_contract(contract_path)
    if contract.backend != "ultralytics":
        raise SystemExit(f"Expected run contract backend 'ultralytics'; got {contract.backend!r}")
    ledger_task = contract.task
    if ledger_task not in LEDGER_TASKS:
        raise SystemExit(f"task must be one of {sorted(LEDGER_TASKS)}; got {ledger_task!r}")
    hp = dict(contract.training.hyperparameters)
    ut_raw = hp.pop("ultralytics_train", {})
    ub_raw = hp.pop("ultralytics_bridge", {})
    ultralytics_train = _coerce_ultralytics_train_block(
        ut_raw if isinstance(ut_raw, dict) else {}, contract_path
    )
    bridge = _coerce_ultralytics_bridge(ub_raw if isinstance(ub_raw, dict) else {}, contract_path)
    flat: dict[str, Any] = dict(hp)
    flat["model_name_or_path"] = contract.model.model_id
    flat["dataset_name"] = contract.dataset.identifier
    if contract.dataset.config_name is not None:
        flat["dataset_config_name"] = contract.dataset.config_name
    flat["dataset_revision"] = contract.dataset.revision
    roles = dict(contract.dataset.column_mapping)
    if ledger_task == "classify_yolo" and "label" in roles:
        flat["label_column"] = roles["label"]
    hints = dict(contract.model.architecture_hints)
    if hints.get("objects_category_field") is not None:
        flat["objects_category_field"] = hints["objects_category_field"]
    parser = HfArgumentParser(cast(Any, (ModelArguments, YoloDataArguments, TrainingArguments)))
    model_args, data_args, training_args = parser.parse_dict(flat, allow_extra_keys=True)

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

    out_base = _require_training_output_dir(training_args)
    out_base.mkdir(parents=True, exist_ok=True)
    trackio.init(project=str(out_base), name=training_args.run_name)

    if ledger_task in ("detect_yolo", "track_yolo"):
        run_detect_family(
            ledger_task,
            model_args,
            data_args,
            training_args,
            detect_label_lines,
            start_time,
            ultralytics_train,
            bridge,
        )
    elif ledger_task == "pose_yolo":
        run_detect_family(
            ledger_task,
            model_args,
            data_args,
            training_args,
            pose_label_lines,
            start_time,
            ultralytics_train,
            bridge,
        )
    elif ledger_task == "obb_yolo":
        run_detect_family(
            ledger_task,
            model_args,
            data_args,
            training_args,
            obb_label_lines,
            start_time,
            ultralytics_train,
            bridge,
        )
    elif ledger_task == "segment_yolo":
        run_segment_yolo(
            model_args, data_args, training_args, start_time, ultralytics_train, bridge
        )
    elif ledger_task == "classify_yolo":
        run_classify_yolo(
            model_args, data_args, training_args, start_time, ultralytics_train, bridge
        )
    else:
        raise SystemExit(f"Unhandled task_type: {ledger_task}")


def main() -> None:
    start_time = time.time()
    if len(sys.argv) != 2:
        raise SystemExit("Usage: train_ultralytics.py <run-contract.yaml|run-contract.json>")
    contract_path = Path(os.path.abspath(sys.argv[1]))
    if not contract_path.is_file():
        raise SystemExit(f"Run contract path is not a file: {contract_path}")
    _run_ultralytics_from_contract(contract_path, start_time=start_time)


if __name__ == "__main__":
    main()
