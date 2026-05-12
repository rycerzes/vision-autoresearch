"""Training utilities — argument mapping, metric parsing, summary emission.

Shared between HF and Ultralytics backends.  Maps universal config args
to backend-native equivalents.
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ══ Universal training arguments ════════════════════════════════


@dataclass
class UniversalTrainingArgs:
    """Backend-agnostic training arguments parsed from YAML config.

    These are the YAML surface.  Each backend maps them to its own
    native equivalents (HF TrainingArguments / Ultralytics train kwargs).
    """

    # Model & data
    model_name_or_path: str = ""
    dataset_name: str = ""
    dataset_config_name: str | None = None

    # Training
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    seed: int = 42
    fp16: bool = True
    bf16: bool = False
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Image
    image_size: int = 640

    # Evaluation
    eval_strategy: str = "epoch"
    eval_steps: int | None = None
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str | None = None

    # Output
    output_dir: str = "./output"
    logging_steps: int = 50
    dataloader_num_workers: int = 4
    run_name: str | None = None

    # LR scheduler (HF-specific but universally accepted)
    lr_scheduler_type: str = "cosine"

    # Backend-specific overrides (pass-through)
    hf_train: dict[str, Any] = field(default_factory=dict)
    ultralytics_train: dict[str, Any] = field(default_factory=dict)
    ultralytics_bridge: dict[str, Any] = field(default_factory=dict)

    # Column map override
    column_map: dict[str, Any] | None = None

    # Head category override
    head_category: str | None = None

    # Promotion config
    promotion_metric: str | None = None
    promotion_direction: str | None = None

    # Research mode
    modification_module: str | None = None  # path to modification.py

    # Data options
    train_val_split: float = 0.15
    max_train_samples: int | None = None
    max_eval_samples: int | None = None

    # Freeze
    freeze_backbone: bool = False

    # Augmentation
    use_albumentations: bool = True
    use_trivial_augment: bool = False

    # Trust remote code
    trust_remote_code: bool = True


def parse_config_yaml(yaml_path: str) -> UniversalTrainingArgs:
    """Parse a YAML config file into UniversalTrainingArgs."""
    import yaml

    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}

    args = UniversalTrainingArgs()
    for key, value in raw.items():
        if hasattr(args, key) and value is not None:
            setattr(args, key, value)

    return args


# ══ Backend arg mapping ═════════════════════════════════════════


def to_hf_training_args(args: UniversalTrainingArgs) -> dict[str, Any]:
    """Map universal args → HF TrainingArguments kwargs.

    Returns a dict suitable for ``TrainingArguments(**result)``.
    Backend-specific ``hf_train`` overrides are merged last.
    """
    result: dict[str, Any] = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_train_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "warmup_ratio": args.warmup_ratio,
        "seed": args.seed,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_grad_norm": args.max_grad_norm,
        "eval_strategy": args.eval_strategy,
        "save_strategy": args.save_strategy,
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": args.load_best_model_at_end,
        "logging_steps": args.logging_steps,
        "dataloader_num_workers": args.dataloader_num_workers,
        "lr_scheduler_type": args.lr_scheduler_type,
        "remove_unused_columns": False,  # we handle columns ourselves
        "do_train": True,
        "do_eval": True,
    }

    if args.run_name:
        result["run_name"] = args.run_name
    if args.metric_for_best_model:
        result["metric_for_best_model"] = args.metric_for_best_model
    if args.eval_steps:
        result["eval_steps"] = args.eval_steps

    # Merge HF-specific overrides
    result.update(args.hf_train)
    return result


def to_ultralytics_train_kwargs(
    args: UniversalTrainingArgs,
    data_yaml: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Map universal args → Ultralytics .train() kwargs.

    Universal args map:
        num_train_epochs → epochs
        learning_rate → lr0
        per_device_train_batch_size → batch
        image_size → imgsz
        fp16 → amp
        dataloader_num_workers → workers
        seed → seed

    Backend-specific ``ultralytics_train`` overrides are merged last.
    """
    # Start with defaults from universal args
    result: dict[str, Any] = {
        "epochs": args.num_train_epochs,
        "lr0": args.learning_rate,
        "batch": args.per_device_train_batch_size,
        "imgsz": args.image_size,
        "amp": args.fp16 or args.bf16,
        "workers": args.dataloader_num_workers,
        "seed": args.seed,
        "weight_decay": args.weight_decay,
    }

    # Map lr_scheduler_type
    if args.lr_scheduler_type == "cosine":
        result["cos_lr"] = True

    # Script-owned keys
    result["data"] = str(data_yaml)
    result["project"] = str(output_dir)
    result["name"] = "ultralytics"
    result["exist_ok"] = True
    result["verbose"] = True

    # Merge ultralytics-specific overrides (user overrides universal defaults)
    user_ultra = dict(args.ultralytics_train)

    # Pop trainer (handled separately)
    user_ultra.pop("trainer", None)

    # Reserved keys that we always control
    for reserved in ("data", "project", "name", "exist_ok"):
        user_ultra.pop(reserved, None)

    result.update(user_ultra)
    return result


# ══ Ultralytics metric parsing ══════════════════════════════════


def pick_csv_metric(last_row: dict[str, str], keys: tuple[str, ...]) -> float:
    """Pick the first available metric value from a CSV row."""
    for k in keys:
        raw = last_row.get(k)
        if raw in (None, ""):
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return 0.0


def read_ultralytics_train_metrics(run_dir: Path) -> dict[str, Any]:
    """Read train loss and epoch from Ultralytics results.csv."""
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
        lk = key.lower().strip()
        if "train" in lk and "loss" in lk:
            try:
                train_loss = float(last[key])
                break
            except (TypeError, ValueError):
                continue
    epoch = 0
    if "                   epoch" in last:
        try:
            epoch = int(float(last["                   epoch"]))
        except (TypeError, ValueError):
            pass
    elif "epoch" in last:
        try:
            epoch = int(float(last["epoch"]))
        except (TypeError, ValueError):
            pass
    return {"train_loss": train_loss, "epoch": epoch}


def read_ultralytics_eval_metrics(
    run_dir: Path, head_category: str
) -> dict[str, float]:
    """Read eval metrics from Ultralytics results.csv.

    Maps Ultralytics metric column names to standard metric keys.
    """
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    # Strip whitespace from keys (Ultralytics pads with spaces)
    last = {k.strip(): v for k, v in rows[-1].items()}

    if head_category == "classification":
        acc = pick_csv_metric(
            last,
            ("metrics/accuracy_top1", "metrics/accuracy_top1(top1)", "accuracy/top1"),
        )
        return {"accuracy": acc}

    if head_category == "detection":
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

    if head_category == "structured_detection":
        # Pose metrics
        return {
            "oks_map": pick_csv_metric(
                last,
                ("metrics/mAP50-95(P)", "metrics/mAP50-95(B)"),
            ),
            "mAP_50": pick_csv_metric(
                last,
                ("metrics/mAP50(P)", "metrics/mAP50(B)"),
            ),
        }

    # Fallback: return all metrics/ columns as-is
    result: dict[str, float] = {}
    for k, v in last.items():
        if k.startswith("metrics/"):
            try:
                short_key = k.replace("metrics/", "").replace("(B)", "").replace("(M)", "").replace("(P)", "")
                result[short_key] = float(v)
            except (TypeError, ValueError):
                pass
    return result


# ══ Summary emission ════════════════════════════════════════════


def emit_summary(
    head_category: str,
    eval_metrics: dict[str, Any],
    train_metrics: dict[str, Any],
    training_seconds: float,
    peak_vram_mb: float,
) -> None:
    """Print structured summary block for parse_metric.py extraction."""
    from engine.metrics import HEAD_METRICS

    print("\n--- VISION AUTORESEARCH SUMMARY ---")
    print(f"head_category: {head_category}")

    # Emit known metrics in canonical order first
    emitted: set[str] = set()
    canonical_metrics = HEAD_METRICS.get(head_category, [])
    for key in canonical_metrics:
        if key in eval_metrics:
            print(f"{key}: {eval_metrics[key]}")
            emitted.add(key)

    # Emit remaining eval metrics
    for key, val in sorted(eval_metrics.items()):
        if key not in emitted:
            print(f"{key}: {val}")

    print(f"training_seconds: {training_seconds:.1f}")
    print(f"peak_vram_mb: {peak_vram_mb:.0f}")
    print(f"train_loss: {train_metrics.get('train_loss', 0.0)}")
    print(f"num_train_epochs: {train_metrics.get('epoch', train_metrics.get('num_train_epochs', 0))}")
    print("--- END SUMMARY ---")


# ══ Ultralytics trainer resolution ═════════════════════════════


def resolve_ultralytics_trainer(
    model: Any,
    args: UniversalTrainingArgs,
) -> type | None:
    """Resolve the Ultralytics trainer class from config.

    Checks (in order):
    1. Explicit ``ultralytics_train.trainer`` string
    2. ``ultralytics_bridge.trainer`` string
    3. Auto-resolution for YOLOE/World models
    4. None → use default trainer
    """
    # Explicit trainer from ultralytics_train
    trainer_name = args.ultralytics_train.get("trainer")
    if not trainer_name:
        trainer_name = args.ultralytics_bridge.get("trainer")

    if isinstance(trainer_name, str) and trainer_name.strip():
        return _resolve_trainer_class(trainer_name.strip())

    # Auto-resolve for YOLOE
    model_name = args.model_name_or_path.lower()
    if "yoloe" in model_name:
        return _auto_resolve_yoloe_trainer(args)

    return None


def _resolve_trainer_class(name: str) -> type:
    """Look up a trainer class by name from Ultralytics internals."""
    import importlib

    # Common trainer locations
    search_modules = [
        "ultralytics.models.yolo.detect",
        "ultralytics.models.yolo.segment",
        "ultralytics.models.yolo.classify",
        "ultralytics.models.yolo.pose",
        "ultralytics.models.yolo.obb",
        "ultralytics.models.yolo.world",
        "ultralytics.models.yoloe",
    ]

    for mod_path in search_modules:
        try:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, name, None)
            if cls is not None:
                return cls
        except (ImportError, AttributeError):
            continue

    raise ValueError(
        f"Cannot resolve Ultralytics trainer class: {name!r}. "
        f"Searched in: {search_modules}"
    )


def _auto_resolve_yoloe_trainer(args: UniversalTrainingArgs) -> type | None:
    """Auto-select YOLOE trainer variant."""
    mode = args.ultralytics_bridge.get("yoloe_training", "full")
    task = getattr(args, "_ultralytics_task", "detect")

    try:
        import ultralytics.models.yoloe as yoloe_mod
    except ImportError:
        return None

    if "segment" in task or "seg" in task:
        if mode == "linear_probe":
            return getattr(yoloe_mod, "YOLOEPESegTrainer", None)
        return getattr(yoloe_mod, "YOLOESegTrainer", None)
    else:
        if mode == "linear_probe":
            return getattr(yoloe_mod, "YOLOEPEFreeTrainer", None)
        return getattr(yoloe_mod, "YOLOETrainer", None)
