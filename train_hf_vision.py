"""Shared Hugging Face vision trainer entrypoint (model loaders + adaptation modes).

Stable infrastructure — experiments modify config YAMLs only.

Configs use ``model_loader`` (``auto_task_head`` | ``auto_model`` | ``auto_backbone``) and
``adaptation_mode`` (full fine-tune, frozen backbone / linear probe, eval-only, …). Task
``classify`` uses ``vision_lab.hf_vision.loaders.load_hf_vision_model``; ``detect`` and
``segment`` delegate to ``vision_lab.hf_vision.detect_train`` / ``segment_train`` (shared Hub /
Trackio / logging / summary plumbing in ``vision_lab.hf_vision.runner_session`` and
``summary_block``).
"""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import evaluate
import numpy as np
import torch
import transformers
import yaml
from datasets import load_dataset
from transformers import DefaultDataCollator, HfArgumentParser, Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction

ROOT = Path(__file__).resolve().parent
_SCRIPTS_DIR = ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from vision_lab.hf_vision import apply_adaptation_mode, build_transforms, load_hf_vision_model
from vision_lab.hf_vision.constants import (
    ADAPTATION_MODE_CHOICES,
    HF_VISION_SUPPORTED_TASKS,
    MODEL_LOADER_CHOICES,
    ROUTED_TASK_IDS,
)
from vision_lab.hf_vision.runner_session import finish_trackio_session, setup_hf_training_environment
from vision_lab.hf_vision.summary_block import print_vision_autoresearch_summary

logger = logging.getLogger(__name__)


def _load_raw_config(config_path: Path) -> dict[str, Any]:
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    return data if isinstance(data, dict) else {}


def _delegate_detect_train(config_path: Path) -> None:
    saved = sys.argv[:]
    try:
        sys.argv = [saved[0] if saved else "train_hf_vision", str(config_path)]
        from vision_lab.hf_vision.detect_train import main as detect_main

        detect_main()
    finally:
        sys.argv = saved


def _delegate_segment_train(config_path: Path) -> None:
    saved = sys.argv[:]
    try:
        sys.argv = [saved[0] if saved else "train_hf_vision", str(config_path)]
        from vision_lab.hf_vision.segment_train import main as segment_main

        segment_main()
    finally:
        sys.argv = saved


@dataclass
class TaskArguments:
    """Which benchmark contract / summary this run uses."""

    task_type: str = field(
        default="classify",
        metadata={"help": "Task id (must be classify when using the classification argument bundle)."},
    )


@dataclass
class ModelArguments:
    """Checkpoint + ``model_loader`` dispatch."""

    model_name_or_path: str = field(
        default="google/vit-base-patch16-224",
        metadata={"help": "Pretrained model identifier or checkpoint directory."},
    )
    model_loader: str = field(
        default="auto_task_head",
        metadata={
            "help": (
                "Weight graph: auto_task_head (AutoModelFor*), auto_model (AutoModel + probe head), "
                "auto_backbone (AutoBackbone + probe head)."
            )
        },
    )
    config_name: str | None = field(
        default=None,
        metadata={"help": "Optional config id or path (defaults to model_name_or_path)."},
    )
    cache_dir: str | None = field(default=None, metadata={"help": "HF cache directory."})
    model_revision: str = field(default="main", metadata={"help": "Revision on the Hub."})
    image_processor_name: str | None = field(
        default=None,
        metadata={"help": "Optional processor id or path."},
    )
    ignore_mismatched_sizes: bool = field(
        default=True,
        metadata={"help": "Allow head resize when num_labels differs from the checkpoint."},
    )
    token: str | None = field(default=None, metadata={"help": "Hub token for private assets."})
    trust_remote_code: bool = field(default=False, metadata={"help": "Trust remote code on load."})


@dataclass
class DataArguments:
    """Dataset columns and optional subsampling (classification path)."""

    dataset_name: str = field(default="ethz/food101", metadata={"help": "HF dataset id."})
    dataset_config_name: str | None = field(default=None, metadata={"help": "HF config name."})
    train_val_split: float | None = field(
        default=0.15,
        metadata={"help": "Fraction held out from train when no validation split exists."},
    )
    max_train_samples: int | None = field(default=None, metadata={"help": "Cap train rows (debug)."})
    max_eval_samples: int | None = field(default=None, metadata={"help": "Cap eval rows (debug)."})
    image_column_name: str = field(default="image", metadata={"help": "Image column."})
    label_column_name: str = field(default="label", metadata={"help": "Label column."})


@dataclass
class AdaptationArguments:
    """How trainable weights are chosen after load (orthogonal to ``TrainingArguments``)."""

    adaptation_mode: str = field(
        default="full_finetune",
        metadata={
            "help": "One of: "
            + ", ".join(sorted(ADAPTATION_MODE_CHOICES))
            + "."
        },
    )


def main() -> None:
    if len(sys.argv) != 2 or not sys.argv[1].endswith((".yaml", ".yml", ".json")):
        raise SystemExit("Usage: train_hf_vision.py <config.yaml|config.json>")

    cfg_path = Path(os.path.abspath(sys.argv[1]))
    raw = _load_raw_config(cfg_path)
    task = str(raw.get("task_type", "")).strip()
    if task not in ROUTED_TASK_IDS:
        raise SystemExit(
            f"Unsupported task_type={task!r} for train_hf_vision (expected one of {sorted(ROUTED_TASK_IDS)})."
        )
    if task == "detect":
        _delegate_detect_train(cfg_path)
        return
    if task == "segment":
        _delegate_segment_train(cfg_path)
        return

    start_time = time.time()
    parser = HfArgumentParser(
        cast(
            Any,
            (TaskArguments, ModelArguments, DataArguments, AdaptationArguments, TrainingArguments),
        )
    )

    if cfg_path.suffix.lower() in (".yaml", ".yml"):
        task_args, model_args, data_args, adaptation_args, training_args = parser.parse_yaml_file(
            yaml_file=str(cfg_path),
            allow_extra_keys=True,
        )
    elif cfg_path.suffix.lower() == ".json":
        task_args, model_args, data_args, adaptation_args, training_args = parser.parse_json_file(
            json_file=str(cfg_path)
        )
    else:
        raise SystemExit("Config must be .yaml, .yml, or .json")

    if str(task_args.task_type).strip() != "classify":
        raise SystemExit(f"Expected task_type=classify in config, got {task_args.task_type!r}")

    adaptation_mode = adaptation_args.adaptation_mode.strip()

    if adaptation_mode not in ADAPTATION_MODE_CHOICES:
        raise SystemExit(
            f"Invalid adaptation_mode={adaptation_mode!r}; "
            f"expected one of {sorted(ADAPTATION_MODE_CHOICES)}."
        )

    ml = model_args.model_loader.strip()
    if ml not in MODEL_LOADER_CHOICES:
        raise SystemExit(f"Invalid model_loader={model_args.model_loader!r}; expected {sorted(MODEL_LOADER_CHOICES)}.")

    if task_args.task_type not in HF_VISION_SUPPORTED_TASKS:
        raise SystemExit(
            f"Internal error: classify must be in HF_VISION_SUPPORTED_TASKS ({sorted(HF_VISION_SUPPORTED_TASKS)})."
        )

    setup_hf_training_environment(training_args, logger=logger)

    logger.info(
        "HF vision runner: task=%s model_loader=%s adaptation_mode=%s",
        task_args.task_type,
        ml,
        adaptation_mode,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    _run_classify(
        task_args.task_type,
        model_args,
        data_args,
        adaptation_mode,
        training_args,
        start_time,
    )


def _run_classify(
    task_type: str,
    model_args: ModelArguments,
    data_args: DataArguments,
    adaptation_mode: str,
    training_args: TrainingArguments,
    start_time: float,
) -> None:
    dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )

    label_col = data_args.label_column_name
    if label_col not in dataset["train"].column_names:
        candidates = [c for c in dataset["train"].column_names if c in ("label", "labels", "class", "fine_label")]
        if candidates:
            label_col = candidates[0]
            logger.info("Label column %r missing; using %r", data_args.label_column_name, label_col)
        else:
            raise ValueError(
                f"Label column {data_args.label_column_name!r} not found. "
                f"Available: {dataset['train'].column_names}"
            )

    label_feature = dataset["train"].features[label_col]
    if hasattr(label_feature, "names"):
        label_names = label_feature.names
    else:
        unique_labels = sorted(set(dataset["train"][label_col]))
        if all(isinstance(l, str) for l in unique_labels):  # noqa: E741
            label_names = unique_labels
        else:
            label_names = [str(l) for l in unique_labels]  # noqa: E741

    num_labels = len(label_names)
    id2label = dict(enumerate(label_names))
    label2id = {v: k for k, v in id2label.items()}
    logger.info("Number of classes: %s", num_labels)

    sample_label = dataset["train"][0][label_col]
    if isinstance(sample_label, str):
        for split_name in list(dataset.keys()):
            dataset[split_name] = dataset[split_name].map(
                lambda ex, lc=label_col: {lc: label2id[ex[lc]]},
            )

    dataset["train"] = dataset["train"].shuffle(seed=training_args.seed)

    tvs = None if "validation" in dataset else data_args.train_val_split
    if isinstance(tvs, float) and tvs and tvs > 0.0:
        split = dataset["train"].train_test_split(tvs, seed=training_args.seed)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    if data_args.max_train_samples is not None:
        max_train = min(data_args.max_train_samples, len(dataset["train"]))
        dataset["train"] = dataset["train"].select(range(max_train))
        logger.info("Truncated training set to %s samples", max_train)
    if data_args.max_eval_samples is not None and "validation" in dataset:
        max_eval = min(data_args.max_eval_samples, len(dataset["validation"]))
        dataset["validation"] = dataset["validation"].select(range(max_eval))
        logger.info("Truncated validation set to %s samples", max_eval)

    model, image_processor = load_hf_vision_model(
        task_type=task_type,
        model_loader=model_args.model_loader,
        model_name_or_path=model_args.model_name_or_path,
        config_name=model_args.config_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        cache_dir=model_args.cache_dir,
        model_revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        image_processor_name=model_args.image_processor_name,
    )

    apply_adaptation_mode(model, adaptation_mode, architecture="classify")

    train_transforms = build_transforms(image_processor, is_training=True)
    val_transforms = build_transforms(image_processor, is_training=False)
    image_col = data_args.image_column_name

    def preprocess_train(examples: dict[str, Any]) -> dict[str, Any]:
        return {
            "pixel_values": [train_transforms(img.convert("RGB")) for img in examples[image_col]],
            "labels": examples[label_col],
        }

    def preprocess_val(examples: dict[str, Any]) -> dict[str, Any]:
        return {
            "pixel_values": [val_transforms(img.convert("RGB")) for img in examples[image_col]],
            "labels": examples[label_col],
        }

    dataset["train"].set_transform(preprocess_train)
    if "validation" in dataset:
        dataset["validation"].set_transform(preprocess_val)
    if "test" in dataset:
        dataset["test"].set_transform(preprocess_val)

    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred: EvalPrediction) -> dict[str, Any]:
        predictions = np.argmax(eval_pred.predictions, axis=1)
        computed = accuracy_metric.compute(
            predictions=predictions,
            references=eval_pred.label_ids,
        )
        return cast(dict[str, Any], computed)

    eval_dataset = None
    if training_args.do_eval:
        if "validation" in dataset:
            eval_dataset = dataset["validation"]
        elif "test" in dataset:
            eval_dataset = dataset["test"]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=eval_dataset,
        processing_class=image_processor,
        data_collator=DefaultDataCollator(),
        compute_metrics=compute_metrics,
    )

    train_metrics: dict[str, Any] = {}
    if training_args.do_train:
        if adaptation_mode in ("feature_extract_eval", "zero_shot_eval"):
            logger.warning("do_train=True with eval-only adaptation; skipping trainer.train().")
        else:
            train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
            trainer.save_model()
            train_metrics = train_result.metrics
            trainer.log_metrics("train", train_metrics)
            trainer.save_metrics("train", train_metrics)
            trainer.save_state()

    eval_metrics: dict[str, Any] = {}
    if training_args.do_eval:
        test_dataset = dataset.get("test", dataset.get("validation"))
        test_prefix = "test" if "test" in dataset else "eval"
        if test_dataset is not None:
            model.eval()
            eval_metrics = trainer.evaluate(
                eval_dataset=cast(Any, test_dataset),
                metric_key_prefix=test_prefix,
            )
            trainer.log_metrics(test_prefix, eval_metrics)
            trainer.save_metrics(test_prefix, eval_metrics)

    training_seconds = time.time() - start_time
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0

    finish_trackio_session()
    print_vision_autoresearch_summary(
        task_type, eval_metrics, train_metrics, training_seconds, peak_vram_mb
    )

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": data_args.dataset_name,
        "tags": ["image-classification", "vision", "hf-vision-runner"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
