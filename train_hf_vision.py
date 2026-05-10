"""Shared Hugging Face vision trainer entrypoint (model loaders + adaptation modes).

Stable infrastructure — the trainer entry is a validated ``RunContract`` file only:

  ``train_hf_vision.py <run-contract.yaml|run-contract.json>``

Configs use ``model_loader`` (``auto_task_head`` | ``auto_model`` | ``auto_backbone``) and
``adaptation_mode`` (full fine-tune, frozen backbone / linear probe, eval-only, …). Supported
HF tasks are loaded through ``vision_lab.hf_vision.loaders.load_hf_vision_model`` so
task-head Transformers models are the forward-only training path.
"""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, cast

import albumentations as A
import evaluate
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import DefaultDataCollator, HfArgumentParser, Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction

ROOT = Path(__file__).resolve().parent
_SCRIPTS_DIR = ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from vision_lab.hf_vision import apply_adaptation_mode, build_transforms, load_hf_vision_model
from vision_lab.hf_vision.constants import (
    ADAPTATION_MODE_CHOICES,
    ROUTED_TASK_IDS,
)
from vision_lab.hf_vision.runner_session import (
    finish_trackio_session,
    setup_hf_training_environment,
)
from vision_lab.hf_vision.summary_block import print_vision_autoresearch_summary

logger = logging.getLogger(__name__)


@dataclass
class TaskArguments:
    """Which benchmark contract / summary this run uses."""

    task_type: str = field(
        default="classify",
        metadata={"help": f"Task id (one of {sorted(ROUTED_TASK_IDS)})."},
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
    dataset_revision: str | None = field(
        default=None,
        metadata={"help": "Optional Hub dataset revision (commit, tag, or branch)."},
    )
    train_val_split: float | None = field(
        default=0.15,
        metadata={"help": "Fraction held out from train when no validation split exists."},
    )
    max_train_samples: int | None = field(
        default=None, metadata={"help": "Cap train rows (debug)."}
    )
    max_eval_samples: int | None = field(default=None, metadata={"help": "Cap eval rows (debug)."})
    image_column_name: str = field(default="image", metadata={"help": "Image column."})
    label_column_name: str = field(default="label", metadata={"help": "Label column."})
    mask_column_name: str = field(default="mask", metadata={"help": "Segmentation mask column."})
    annotation_column_name: str = field(
        default="annotation",
        metadata={
            "help": "Instance/panoptic annotation column containing semantic and instance ids."
        },
    )
    image_height: int | None = field(
        default=512, metadata={"help": "Dense segmentation image height."}
    )
    image_width: int | None = field(
        default=512, metadata={"help": "Dense segmentation image width."}
    )
    do_reduce_labels: bool = field(
        default=False, metadata={"help": "Reduce background label 0 when supported."}
    )


@dataclass
class AdaptationArguments:
    """How trainable weights are chosen after load (orthogonal to ``TrainingArguments``)."""

    adaptation_mode: str = field(
        default="full_finetune",
        metadata={"help": "One of: " + ", ".join(sorted(ADAPTATION_MODE_CHOICES)) + "."},
    )


def _dispatch_hf_trainer_contract(contract_path: Path) -> None:
    """Execute a resolved ``RunContract`` (``hf_trainer`` backend) without YAML task routing."""
    from vision_lab.contracts.loader import load_run_contract
    from vision_lab.hf_vision.detect_train import (
        DataTrainingArguments as DetDataArguments,
    )
    from vision_lab.hf_vision.detect_train import (
        ModelArguments as DetModelArguments,
    )
    from vision_lab.hf_vision.detect_train import (
        run_detect_training,
    )
    from vision_lab.hf_vision.segment_train import (
        DataTrainingArguments as SegDataArguments,
    )
    from vision_lab.hf_vision.segment_train import (
        ModelArguments as SegModelArguments,
    )
    from vision_lab.hf_vision.segment_train import (
        run_segment_training,
    )

    contract = load_run_contract(contract_path)
    if contract.backend != "hf_trainer":
        raise SystemExit(f"Expected run contract backend 'hf_trainer'; got {contract.backend!r}")
    roles = dict(contract.dataset.column_mapping)
    hp = dict(contract.training.hyperparameters)
    tr_parser = HfArgumentParser(cast(Any, (TrainingArguments,)))
    training_args, = tr_parser.parse_dict(hp, allow_extra_keys=True)

    task = contract.task
    hints = dict(contract.model.architecture_hints)

    if task == "classify":
        if "image" not in roles or "label" not in roles:
            raise SystemExit("classify run-contract requires column_mapping roles 'image' and 'label'.")
        model_args = ModelArguments(
            model_name_or_path=contract.model.model_id,
            model_loader=contract.model.loader_strategy,
            config_name=hints.get("config_name") if hints.get("config_name") is not None else None,
            cache_dir=str(hints["cache_dir"]) if hints.get("cache_dir") is not None else None,
            model_revision=str(hints.get("model_revision", "main")),
            image_processor_name=hints.get("image_processor_name"),
            ignore_mismatched_sizes=bool(hints.get("ignore_mismatched_sizes", True)),
            token=str(hints["token"]) if hints.get("token") is not None else None,
            trust_remote_code=bool(hints.get("trust_remote_code", False)),
        )
        adaptation_mode = str(hints.get("adaptation_mode", "full_finetune")).strip()
        data_args = DataArguments(
            dataset_name=contract.dataset.identifier,
            dataset_config_name=contract.dataset.config_name,
            dataset_revision=contract.dataset.revision,
            image_column_name=roles["image"],
            label_column_name=roles["label"],
        )
        start_time = time.time()
        setup_hf_training_environment(training_args, logger=logger)
        _run_classify(
            task,
            model_args,
            data_args,
            adaptation_mode,
            training_args,
            start_time,
            strict_columns=True,
        )
        return

    if task == "detect":
        if "image" not in roles or "objects" not in roles:
            raise SystemExit("detect run-contract requires column_mapping roles 'image' and 'objects'.")
        flat: dict[str, Any] = dict(hp)
        flat.setdefault("model_name_or_path", contract.model.model_id)
        flat.setdefault("model_loader", contract.model.loader_strategy)
        flat.setdefault("dataset_name", contract.dataset.identifier)
        if contract.dataset.config_name is not None:
            flat["dataset_config_name"] = contract.dataset.config_name
        flat["dataset_revision"] = contract.dataset.revision
        flat["image_column_name"] = roles["image"]
        flat["objects_column_name"] = roles["objects"]
        if hints.get("adaptation_mode") is not None:
            flat["adaptation_mode"] = str(hints["adaptation_mode"])
        dparser = HfArgumentParser(cast(Any, (DetModelArguments, DetDataArguments, TrainingArguments)))
        det_model, det_data, det_train = dparser.parse_dict(flat, allow_extra_keys=True)
        run_detect_training(det_model, det_data, det_train)
        return

    if task == "segment":
        if "image" not in roles or "mask" not in roles:
            raise SystemExit("segment run-contract requires column_mapping roles 'image' and 'mask'.")
        flat = dict(hp)
        flat.setdefault("model_name_or_path", contract.model.model_id)
        flat.setdefault("model_loader", contract.model.loader_strategy)
        flat.setdefault("dataset_name", contract.dataset.identifier)
        if contract.dataset.config_name is not None:
            flat["dataset_config_name"] = contract.dataset.config_name
        flat["dataset_revision"] = contract.dataset.revision
        flat["image_column_name"] = roles["image"]
        flat["mask_column_name"] = roles["mask"]
        if hints.get("prompt_type") is not None:
            flat["prompt_type"] = str(hints["prompt_type"])
        if hints.get("prompt_column_name") is not None:
            flat["prompt_column_name"] = hints["prompt_column_name"]
        if hints.get("bbox_column_name") is not None:
            flat["bbox_column_name"] = hints["bbox_column_name"]
        if hints.get("point_column_name") is not None:
            flat["point_column_name"] = hints["point_column_name"]
        if hints.get("adaptation_mode") is not None:
            flat["adaptation_mode"] = str(hints["adaptation_mode"])
        sparser = HfArgumentParser(cast(Any, (SegModelArguments, SegDataArguments, TrainingArguments)))
        seg_model, seg_data, seg_train = sparser.parse_dict(flat, allow_extra_keys=True)
        run_segment_training(seg_model, seg_data, seg_train)
        return

    if task == "semantic_segment":
        if "image" not in roles or "mask" not in roles:
            raise SystemExit(
                "semantic_segment run-contract requires column_mapping roles 'image' and 'mask'."
            )
        model_args = ModelArguments(
            model_name_or_path=contract.model.model_id,
            model_loader=contract.model.loader_strategy,
            config_name=hints.get("config_name") if hints.get("config_name") is not None else None,
            cache_dir=str(hints["cache_dir"]) if hints.get("cache_dir") is not None else None,
            model_revision=str(hints.get("model_revision", "main")),
            image_processor_name=hints.get("image_processor_name"),
            ignore_mismatched_sizes=bool(hints.get("ignore_mismatched_sizes", True)),
            token=str(hints["token"]) if hints.get("token") is not None else None,
            trust_remote_code=bool(hints.get("trust_remote_code", False)),
        )
        adaptation_mode = str(hints.get("adaptation_mode", "full_finetune")).strip()
        data_args = DataArguments(
            dataset_name=contract.dataset.identifier,
            dataset_config_name=contract.dataset.config_name,
            dataset_revision=contract.dataset.revision,
            image_column_name=roles["image"],
            mask_column_name=roles["mask"],
        )
        start_time = time.time()
        setup_hf_training_environment(training_args, logger=logger)
        _run_semantic_segment(task, model_args, data_args, adaptation_mode, training_args, start_time)
        return

    if task in ("instance_segment", "universal_segment"):
        if "image" not in roles or "annotation" not in roles:
            raise SystemExit(
                f"{task} run-contract requires column_mapping roles 'image' and 'annotation'."
            )
        model_args = ModelArguments(
            model_name_or_path=contract.model.model_id,
            model_loader=contract.model.loader_strategy,
            config_name=hints.get("config_name") if hints.get("config_name") is not None else None,
            cache_dir=str(hints["cache_dir"]) if hints.get("cache_dir") is not None else None,
            model_revision=str(hints.get("model_revision", "main")),
            image_processor_name=hints.get("image_processor_name"),
            ignore_mismatched_sizes=bool(hints.get("ignore_mismatched_sizes", True)),
            token=str(hints["token"]) if hints.get("token") is not None else None,
            trust_remote_code=bool(hints.get("trust_remote_code", False)),
        )
        adaptation_mode = str(hints.get("adaptation_mode", "full_finetune")).strip()
        data_args = DataArguments(
            dataset_name=contract.dataset.identifier,
            dataset_config_name=contract.dataset.config_name,
            dataset_revision=contract.dataset.revision,
            image_column_name=roles["image"],
            annotation_column_name=roles["annotation"],
        )
        start_time = time.time()
        setup_hf_training_environment(training_args, logger=logger)
        _run_dense_instance_or_universal(
            task_type=task,
            model_args=model_args,
            data_args=data_args,
            adaptation_mode=adaptation_mode,
            training_args=training_args,
            start_time=start_time,
            strict_columns=True,
        )
        return

    raise SystemExit(
        f"Unsupported task {task!r} for hf_trainer run contract (supported: {sorted(ROUTED_TASK_IDS)})."
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: train_hf_vision.py <run-contract.yaml|run-contract.json>")
    contract_path = Path(os.path.abspath(sys.argv[1]))
    if not contract_path.is_file():
        raise SystemExit(f"Run contract path is not a file: {contract_path}")
    _dispatch_hf_trainer_contract(contract_path)


def _run_classify(
    task_type: str,
    model_args: ModelArguments,
    data_args: DataArguments,
    adaptation_mode: str,
    training_args: TrainingArguments,
    start_time: float,
    *,
    strict_columns: bool = False,
) -> None:
    dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
        revision=data_args.dataset_revision,
    )

    label_col = data_args.label_column_name
    if strict_columns:
        if label_col not in dataset["train"].column_names:
            raise ValueError(
                f"Label column {label_col!r} not found (strict_columns). "
                f"Available: {dataset['train'].column_names}"
            )
    elif label_col not in dataset["train"].column_names:
        candidates = [
            c
            for c in dataset["train"].column_names
            if c in ("label", "labels", "class", "fine_label")
        ]
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
            train_result = trainer.train(
                resume_from_checkpoint=training_args.resume_from_checkpoint
            )
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


def _mask_to_array(mask: Any) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.int64)


def _discover_semantic_labels(
    dataset: Any,
    mask_col: str,
    max_samples: int = 200,
) -> tuple[dict[int, str], dict[str, int], dict[int, int]]:
    """Discover class ids and build a contiguous id space for training loss.

    Many Hub semantic datasets use sparse/raw ids (e.g. ADE20K-like ids up to 150).
    Segmentation heads are initialized with ``num_labels=len(classes)``, so labels
    must be remapped to ``0..num_labels-1`` before loss computation.
    """
    unique_raw: set[int] = set()
    for idx in range(min(max_samples, len(dataset["train"]))):
        mask = _mask_to_array(dataset["train"][idx][mask_col])
        unique_raw.update(int(v) for v in np.unique(mask) if int(v) >= 0 and int(v) != 255)
    if not unique_raw:
        raise ValueError(f"No non-negative class ids found in mask column {mask_col!r}.")

    raw_ids = sorted(unique_raw)
    raw_to_contiguous = {raw_id: i for i, raw_id in enumerate(raw_ids)}
    id2label = {i: f"class_{raw_id}" for i, raw_id in enumerate(raw_ids)}
    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id, raw_to_contiguous


def _run_semantic_segment(
    task_type: str,
    model_args: ModelArguments,
    data_args: DataArguments,
    adaptation_mode: str,
    training_args: TrainingArguments,
    start_time: float,
) -> None:
    # Keep raw dataset columns (e.g. image/mask) for with_transform() preprocessing.
    # Otherwise Trainer prunes columns before transform and dataloader workers hit KeyError.
    training_args.remove_unused_columns = False

    dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
        revision=data_args.dataset_revision,
    )
    if "train" not in dataset:
        raise ValueError(f"No 'train' split found. Available: {list(dataset.keys())}")
    if "validation" not in dataset and "test" not in dataset:
        dataset["train"] = dataset["train"].shuffle(seed=training_args.seed)
        split = dataset["train"].train_test_split(
            data_args.train_val_split or 0.15, seed=training_args.seed
        )
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    image_col = data_args.image_column_name
    mask_col = data_args.mask_column_name
    if mask_col not in dataset["train"].column_names:
        candidates = [
            c
            for c in (
                "mask",
                "label",
                "annotation",
                "segmentation_mask",
                "semantic_mask",
                "label_map",
            )
            if c in dataset["train"].column_names
        ]
        if candidates:
            mask_col = candidates[0]
            logger.info("Mask column %r missing; using %r", data_args.mask_column_name, mask_col)
        else:
            raise ValueError(
                f"Mask column {data_args.mask_column_name!r} not found. "
                f"Available: {dataset['train'].column_names}"
            )

    id2label, label2id, raw_to_contiguous = _discover_semantic_labels(dataset, mask_col)
    logger.info("Discovered semantic classes: %s", id2label)

    if data_args.max_train_samples is not None:
        max_train = min(data_args.max_train_samples, len(dataset["train"]))
        dataset["train"] = dataset["train"].select(range(max_train))
    eval_key = "validation" if "validation" in dataset else "test"
    if data_args.max_eval_samples is not None and eval_key in dataset:
        max_eval = min(data_args.max_eval_samples, len(dataset[eval_key]))
        dataset[eval_key] = dataset[eval_key].select(range(max_eval))

    model, image_processor = load_hf_vision_model(
        task_type=task_type,
        model_loader=model_args.model_loader,
        model_name_or_path=model_args.model_name_or_path,
        config_name=model_args.config_name,
        num_labels=len(id2label),
        label2id=label2id,
        id2label=id2label,
        cache_dir=model_args.cache_dir,
        model_revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        image_processor_name=model_args.image_processor_name,
    )
    apply_adaptation_mode(model, adaptation_mode, architecture="semantic_segment")

    def transform_batch(examples: dict[str, Any]) -> dict[str, Any]:
        encoded = image_processor(
            [img.convert("RGB") for img in examples[image_col]],
            return_tensors="pt",
        )
        target_size = tuple(encoded["pixel_values"].shape[-2:])
        labels = []
        for raw_mask in examples[mask_col]:
            raw = _mask_to_array(raw_mask).astype(np.int64)
            remapped = np.full(raw.shape, 255, dtype=np.int64)
            for raw_id, contiguous_id in raw_to_contiguous.items():
                remapped[raw == raw_id] = contiguous_id
            mask = torch.as_tensor(remapped, dtype=torch.long)
            if tuple(mask.shape[-2:]) != target_size:
                mask = (
                    F.interpolate(
                        mask.unsqueeze(0).unsqueeze(0).float(),
                        size=target_size,
                        mode="nearest",
                    )
                    .squeeze(0)
                    .squeeze(0)
                    .long()
                )
            labels.append(mask)
        encoded["labels"] = torch.stack(labels)
        return encoded

    dataset["train"] = dataset["train"].with_transform(transform_batch)
    dataset[eval_key] = dataset[eval_key].with_transform(transform_batch)

    def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        logits = (
            eval_pred.predictions[0]
            if isinstance(eval_pred.predictions, tuple)
            else eval_pred.predictions
        )
        preds = np.argmax(logits, axis=1)
        labels = np.asarray(eval_pred.label_ids)
        if preds.shape[-2:] != labels.shape[-2:]:
            resized = F.interpolate(
                torch.as_tensor(preds).unsqueeze(1).float(),
                size=labels.shape[-2:],
                mode="nearest",
            )
            preds = resized.squeeze(1).long().numpy()
        ious = []
        for class_id in id2label:
            pred_mask = preds == class_id
            label_mask = labels == class_id
            union = np.logical_or(pred_mask, label_mask).sum()
            if union == 0:
                continue
            intersection = np.logical_and(pred_mask, label_mask).sum()
            ious.append(float(intersection / union))
        miou = float(np.mean(ious)) if ious else 0.0
        return {"mIoU": round(miou, 4)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset[eval_key] if training_args.do_eval else None,
        processing_class=image_processor,
        data_collator=DefaultDataCollator(),
        compute_metrics=compute_metrics,
    )

    train_metrics: dict[str, Any] = {}
    if training_args.do_train:
        if adaptation_mode in ("feature_extract_eval", "zero_shot_eval"):
            logger.warning("do_train=True with eval-only adaptation; skipping trainer.train().")
        else:
            train_result = trainer.train(
                resume_from_checkpoint=training_args.resume_from_checkpoint
            )
            trainer.save_model()
            train_metrics = train_result.metrics
            trainer.log_metrics("train", train_metrics)
            trainer.save_metrics("train", train_metrics)
            trainer.save_state()

    eval_metrics: dict[str, Any] = {}
    if training_args.do_eval:
        eval_metrics = trainer.evaluate(eval_dataset=cast(Any, dataset[eval_key]))
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    training_seconds = time.time() - start_time
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0

    finish_trackio_session()
    print_vision_autoresearch_summary(
        task_type, eval_metrics, train_metrics, training_seconds, peak_vram_mb
    )

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": data_args.dataset_name,
        "tags": ["semantic-segmentation", "vision", "hf-vision-runner"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


@dataclass
class DenseSegmentationModelOutput:
    class_queries_logits: torch.Tensor
    masks_queries_logits: torch.Tensor


def _nested_cpu(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return type(value)(_nested_cpu(v) for v in value)
    if isinstance(value, Mapping):
        return {k: _nested_cpu(v) for k, v in value.items()}
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def _dense_collate_fn(examples: list[dict[str, Any]]) -> dict[str, Any]:
    batch: dict[str, Any] = {
        "pixel_values": torch.stack([example["pixel_values"] for example in examples]),
        "class_labels": [example["class_labels"] for example in examples],
        "mask_labels": [example["mask_labels"] for example in examples],
    }
    if "pixel_mask" in examples[0]:
        batch["pixel_mask"] = torch.stack([example["pixel_mask"] for example in examples])
    return batch


def _semantic_instance_pair(annotation: Any) -> np.ndarray:
    arr = np.asarray(annotation)
    if arr.ndim == 2:
        return np.stack([arr, arr], axis=-1).astype(np.int64)
    if arr.ndim == 3 and arr.shape[-1] >= 2:
        return arr[..., :2].astype(np.int64)
    if arr.ndim == 3 and arr.shape[0] >= 2:
        return np.moveaxis(arr[:2], 0, -1).astype(np.int64)
    raise ValueError(
        "Dense segmentation annotation must be a 2D id mask or a 2-channel "
        "semantic/instance annotation."
    )


def _dense_label_maps(
    dataset: Any,
    annotation_col: str,
    max_samples: int = 200,
) -> tuple[dict[int, str], dict[str, int], dict[int, int]]:
    first = dataset["train"][0]
    mapping = first.get("semantic_class_to_id")
    if isinstance(mapping, dict) and mapping:
        raw_label2id = {str(k): int(v) for k, v in mapping.items()}
        raw_ids = sorted(set(raw_label2id.values()))  # type: ignore
        raw_to_contiguous = {raw_id: i for i, raw_id in enumerate(raw_ids)}
        id2label = {i: f"class_{raw_id}" for i, raw_id in enumerate(raw_ids)}
        label2id = {v: k for k, v in id2label.items()}
        return id2label, label2id, raw_to_contiguous

    raw_ids: set[int] = set()
    for idx in range(min(max_samples, len(dataset["train"]))):
        ann = _semantic_instance_pair(dataset["train"][idx][annotation_col])
        raw_ids.update(int(v) for v in np.unique(ann[..., 0]) if int(v) >= 0 and int(v) != 255)
    if not raw_ids:
        raise ValueError(f"No semantic class ids found in annotation column {annotation_col!r}.")

    sorted_raw = sorted(raw_ids)
    raw_to_contiguous = {raw_id: i for i, raw_id in enumerate(sorted_raw)}
    id2label = {i: f"class_{raw_id}" for i, raw_id in enumerate(sorted_raw)}
    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id, raw_to_contiguous


def _augment_dense_batch(
    examples: Mapping[str, Any],
    *,
    transform: A.Compose,
    image_processor: Any,
    image_col: str,
    annotation_col: str,
    semantic_id_remap: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    batch: dict[str, Any] = {
        "pixel_values": [],
        "mask_labels": [],
        "class_labels": [],
    }
    for image, annotation in zip(examples[image_col], examples[annotation_col]):
        image_arr = np.asarray(image.convert("RGB"))
        semantic_and_instance = _semantic_instance_pair(annotation)
        output = transform(image=image_arr, mask=semantic_and_instance)
        aug_image = output["image"]
        aug_pair = output["mask"]
        instance_mask = aug_pair[..., 1]
        pairs = np.unique(aug_pair.reshape(-1, 2), axis=0)
        instance_id_to_semantic_id = {}
        for semantic_id, instance_id in pairs:
            sid = int(semantic_id)
            iid = int(instance_id)
            if iid < 0 or sid < 0:
                continue
            if semantic_id_remap is not None:
                if sid not in semantic_id_remap:
                    continue
                sid = int(semantic_id_remap[sid])
            instance_id_to_semantic_id[iid] = sid
        model_inputs = image_processor(
            images=[aug_image],
            segmentation_maps=[instance_mask],
            instance_id_to_semantic_id=instance_id_to_semantic_id,
            return_tensors="pt",
        )
        batch["pixel_values"].append(model_inputs.pixel_values[0])
        batch["mask_labels"].append(model_inputs.mask_labels[0])
        batch["class_labels"].append(model_inputs.class_labels[0])
        if hasattr(model_inputs, "pixel_mask"):
            batch.setdefault("pixel_mask", []).append(model_inputs.pixel_mask[0])
    return batch


class InstanceSegmentationEvaluator:
    def __init__(
        self, image_processor: Any, id2label: Mapping[int, str], threshold: float = 0.0
    ) -> None:
        self.image_processor = image_processor
        self.id2label = id2label
        self.threshold = threshold
        self.metric = MeanAveragePrecision(iou_type="segm", class_metrics=True)

    def _targets(self, target_batch: Any) -> list[dict[str, torch.Tensor]]:
        batch_masks, batch_labels = target_batch[0], target_batch[1]
        return [
            {
                "masks": torch.as_tensor(masks).to(dtype=torch.bool),
                "labels": torch.as_tensor(labels).to(dtype=torch.long),
            }
            for masks, labels in zip(batch_masks, batch_labels)
        ]

    def _predictions(
        self,
        prediction_batch: Any,
        target_sizes: list[tuple[int, int]],
    ) -> list[dict[str, torch.Tensor]]:
        model_output = DenseSegmentationModelOutput(
            class_queries_logits=torch.as_tensor(prediction_batch[0]),
            masks_queries_logits=torch.as_tensor(prediction_batch[1]),
        )
        batch_size = int(model_output.class_queries_logits.shape[0])
        if len(target_sizes) != batch_size:
            logger.warning(
                "Instance eval size mismatch: %d targets for %d predictions; adjusting target_sizes.",
                len(target_sizes),
                batch_size,
            )
            if not target_sizes:
                h, w = model_output.masks_queries_logits.shape[-2:]
                target_sizes = [(int(h), int(w)) for _ in range(batch_size)]
            elif len(target_sizes) < batch_size:
                target_sizes = target_sizes + [target_sizes[-1]] * (batch_size - len(target_sizes))
            else:
                target_sizes = target_sizes[:batch_size]

        post_processed = self.image_processor.post_process_instance_segmentation(
            model_output,
            threshold=self.threshold,
            target_sizes=target_sizes,
            return_binary_maps=True,
        )
        predictions = []
        for image_pred, target_size in zip(post_processed, target_sizes):
            if image_pred["segments_info"]:
                predictions.append(
                    {
                        "masks": image_pred["segmentation"].to(dtype=torch.bool),
                        "labels": torch.tensor(
                            [x["label_id"] for x in image_pred["segments_info"]]
                        ),
                        "scores": torch.tensor([x["score"] for x in image_pred["segments_info"]]),
                    }
                )
            else:
                predictions.append(
                    {
                        "masks": torch.zeros([0, *target_size], dtype=torch.bool),
                        "labels": torch.tensor([], dtype=torch.long),
                        "scores": torch.tensor([], dtype=torch.float),
                    }
                )
        return predictions

    @torch.no_grad()
    def __call__(self, eval_pred: EvalPrediction) -> dict[str, float]:
        prediction_batch = _nested_cpu(eval_pred.predictions)
        target_batch = _nested_cpu(eval_pred.label_ids)
        targets = self._targets(target_batch)
        target_sizes = [(int(t["masks"].shape[-2]), int(t["masks"].shape[-1])) for t in targets]
        predictions = self._predictions(prediction_batch, target_sizes)
        if len(predictions) != len(targets):
            logger.warning(
                "Instance eval count mismatch: %d predictions vs %d targets; trimming to overlap.",
                len(predictions),
                len(targets),
            )
            n = min(len(predictions), len(targets))
            predictions = predictions[:n]
            targets = targets[:n]
        if not predictions or not targets:
            self.metric.reset()
            return {"mask_map": 0.0, "mAP": 0.0, "mAP_50": 0.0}

        self.metric.update(predictions, targets)
        metrics = self.metric.compute()
        self.metric.reset()
        return {
            "mask_map": round(float(metrics.get("map", torch.tensor(0.0)).item()), 4),
            "mAP": round(float(metrics.get("map", torch.tensor(0.0)).item()), 4),
            "mAP_50": round(float(metrics.get("map_50", torch.tensor(0.0)).item()), 4),
        }


class PanopticSegmentationEvaluator:
    def __init__(self, image_processor: Any, threshold: float = 0.0) -> None:
        self.image_processor = image_processor
        self.threshold = threshold

    @staticmethod
    def _match_quality(
        pred_masks: torch.Tensor, target_masks: torch.Tensor
    ) -> tuple[float, float, float]:
        if pred_masks.numel() == 0 and target_masks.numel() == 0:
            return 1.0, 1.0, 1.0
        if pred_masks.numel() == 0 or target_masks.numel() == 0:
            return 0.0, 0.0, 0.0
        pred = pred_masks.to(dtype=torch.bool)
        target = target_masks.to(dtype=torch.bool)
        matched_targets: set[int] = set()
        iou_sum = 0.0
        true_pos = 0
        for pred_mask in pred:
            best_iou = 0.0
            best_idx = -1
            for idx, target_mask in enumerate(target):
                if idx in matched_targets:
                    continue
                intersection = torch.logical_and(pred_mask, target_mask).sum().item()
                union = torch.logical_or(pred_mask, target_mask).sum().item()
                iou = float(intersection / union) if union else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou > 0.5 and best_idx >= 0:
                matched_targets.add(best_idx)
                true_pos += 1
                iou_sum += best_iou
        false_pos = max(int(pred.shape[0]) - true_pos, 0)
        false_neg = max(int(target.shape[0]) - true_pos, 0)
        denom = true_pos + 0.5 * false_pos + 0.5 * false_neg
        pq = iou_sum / denom if denom else 0.0
        sq = iou_sum / true_pos if true_pos else 0.0
        rq = true_pos / denom if denom else 0.0
        return pq, sq, rq

    @torch.no_grad()
    def __call__(self, eval_pred: EvalPrediction) -> dict[str, float]:
        prediction_batch = _nested_cpu(eval_pred.predictions)
        target_batch = _nested_cpu(eval_pred.label_ids)
        targets = [
            {
                "masks": torch.as_tensor(masks).to(dtype=torch.bool),
                "labels": torch.as_tensor(labels).to(dtype=torch.long),
            }
            for masks, labels in zip(target_batch[0], target_batch[1])
        ]
        target_sizes = [tuple(t["masks"].shape[-2:]) for t in targets]
        model_output = DenseSegmentationModelOutput(
            class_queries_logits=torch.as_tensor(prediction_batch[0]),
            masks_queries_logits=torch.as_tensor(prediction_batch[1]),
        )
        batch_size = int(model_output.class_queries_logits.shape[0])
        if len(target_sizes) != batch_size:
            logger.warning(
                "Panoptic eval size mismatch: %d targets for %d predictions; adjusting target_sizes.",
                len(target_sizes),
                batch_size,
            )
            if not target_sizes:
                h, w = model_output.masks_queries_logits.shape[-2:]
                target_sizes = [(int(h), int(w)) for _ in range(batch_size)]
            elif len(target_sizes) < batch_size:
                target_sizes = target_sizes + [target_sizes[-1]] * (batch_size - len(target_sizes))
            else:
                target_sizes = target_sizes[:batch_size]
        if hasattr(self.image_processor, "post_process_panoptic_segmentation"):
            post_processed = self.image_processor.post_process_panoptic_segmentation(
                model_output,
                threshold=self.threshold,
                target_sizes=target_sizes,
            )
        else:
            post_processed = self.image_processor.post_process_instance_segmentation(
                model_output,
                threshold=self.threshold,
                target_sizes=target_sizes,
                return_binary_maps=True,
            )
        scores: list[tuple[float, float, float]] = []
        for image_pred, target in zip(post_processed, targets):
            segmentation = image_pred.get("segmentation")
            segments_info = image_pred.get("segments_info") or []
            pred_masks = []
            if isinstance(segmentation, torch.Tensor) and segments_info:
                for segment in segments_info:
                    sid = segment.get("id")
                    if sid is not None:
                        pred_masks.append(segmentation == sid)
            if pred_masks:
                pred_tensor = torch.stack(pred_masks)
            else:
                pred_tensor = torch.zeros([0, *target["masks"].shape[-2:]], dtype=torch.bool)
            scores.append(self._match_quality(pred_tensor, target["masks"]))
        if not scores:
            return {"pq": 0.0, "sq": 0.0, "rq": 0.0}
        pq = float(np.mean([s[0] for s in scores]))
        sq = float(np.mean([s[1] for s in scores]))
        rq = float(np.mean([s[2] for s in scores]))
        return {"pq": round(pq, 4), "sq": round(sq, 4), "rq": round(rq, 4)}


def _run_dense_instance_or_universal(
    *,
    task_type: str,
    model_args: ModelArguments,
    data_args: DataArguments,
    adaptation_mode: str,
    training_args: TrainingArguments,
    start_time: float,
    strict_columns: bool = False,
) -> None:
    training_args.remove_unused_columns = False
    dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
        revision=data_args.dataset_revision,
    )
    if "train" not in dataset:
        raise ValueError(f"No 'train' split found. Available: {list(dataset.keys())}")
    if "validation" not in dataset and "test" not in dataset:
        dataset["train"] = dataset["train"].shuffle(seed=training_args.seed)
        split = dataset["train"].train_test_split(
            data_args.train_val_split or 0.15, seed=training_args.seed
        )
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    image_col = data_args.image_column_name
    annotation_col = data_args.annotation_column_name
    if strict_columns:
        if annotation_col not in dataset["train"].column_names:
            raise ValueError(
                f"Annotation column {annotation_col!r} not found (strict_columns). "
                f"Available: {dataset['train'].column_names}"
            )
    elif annotation_col not in dataset["train"].column_names:
        candidates = [
            c
            for c in (
                "annotation",
                "panoptic_mask",
                "panoptic_masks",
                "segmentation",
                "mask",
                "label",
            )
            if c in dataset["train"].column_names
        ]
        if candidates:
            annotation_col = candidates[0]
            logger.info(
                "Annotation column %r missing; using %r",
                data_args.annotation_column_name,
                annotation_col,
            )
        else:
            raise ValueError(
                f"Annotation column {data_args.annotation_column_name!r} not found. "
                f"Available: {dataset['train'].column_names}"
            )

    id2label, label2id, raw_to_contiguous = _dense_label_maps(dataset, annotation_col)
    model, image_processor = load_hf_vision_model(
        task_type=task_type,
        model_loader=model_args.model_loader,
        model_name_or_path=model_args.model_name_or_path,
        config_name=model_args.config_name,
        num_labels=len(id2label),
        label2id=label2id,
        id2label=id2label,
        cache_dir=model_args.cache_dir,
        model_revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        image_processor_name=model_args.image_processor_name,
    )
    if data_args.image_height and data_args.image_width:
        image_processor.size = {"height": data_args.image_height, "width": data_args.image_width}
    if hasattr(image_processor, "do_reduce_labels"):
        image_processor.do_reduce_labels = data_args.do_reduce_labels
    if hasattr(image_processor, "reduce_labels"):
        image_processor.reduce_labels = data_args.do_reduce_labels
    apply_adaptation_mode(model, adaptation_mode, architecture="semantic_segment")

    if data_args.max_train_samples is not None:
        dataset["train"] = dataset["train"].select(
            range(min(data_args.max_train_samples, len(dataset["train"])))
        )
    eval_key = "validation" if "validation" in dataset else "test"
    if data_args.max_eval_samples is not None:
        dataset[eval_key] = dataset[eval_key].select(
            range(min(data_args.max_eval_samples, len(dataset[eval_key])))
        )

    train_transform = A.Compose([A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=0.5)])
    val_transform = A.Compose([A.NoOp()])
    train_transform_batch = partial(
        _augment_dense_batch,
        transform=train_transform,
        image_processor=image_processor,
        image_col=image_col,
        annotation_col=annotation_col,
        semantic_id_remap=raw_to_contiguous,
    )
    val_transform_batch = partial(
        _augment_dense_batch,
        transform=val_transform,
        image_processor=image_processor,
        image_col=image_col,
        annotation_col=annotation_col,
        semantic_id_remap=raw_to_contiguous,
    )
    dataset["train"] = dataset["train"].with_transform(train_transform_batch)
    dataset[eval_key] = dataset[eval_key].with_transform(val_transform_batch)

    if task_type == "instance_segment":
        compute_metrics = InstanceSegmentationEvaluator(
            image_processor=image_processor, id2label=id2label
        )
    else:
        compute_metrics = PanopticSegmentationEvaluator(image_processor=image_processor)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset[eval_key] if training_args.do_eval else None,
        processing_class=image_processor,
        data_collator=_dense_collate_fn,
        compute_metrics=compute_metrics,
    )

    train_metrics: dict[str, Any] = {}
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        train_metrics = train_result.metrics
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)
        trainer.save_state()

    eval_metrics: dict[str, Any] = {}
    if training_args.do_eval:
        eval_metrics = trainer.evaluate(eval_dataset=cast(Any, dataset[eval_key]))
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    training_seconds = time.time() - start_time
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
    finish_trackio_session()
    print_vision_autoresearch_summary(
        task_type, eval_metrics, train_metrics, training_seconds, peak_vram_mb
    )

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": data_args.dataset_name,
        "tags": ["image-segmentation", task_type, "vision", "hf-vision-runner"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
