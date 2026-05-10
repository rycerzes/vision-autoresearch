"""Compile an experiment YAML/JSON config into a validated ``RunContract`` file.

Training entrypoints (``train_hf_vision.py``, ``train_ultralytics.py``) accept only a
contract path. Launchers (``hf_job.py``, ``run_local.py``) call this module first when
the user passes a legacy flat config.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from vision_lab.contracts.run_contract import CONTRACT_VERSION, run_contract_to_primitive_dict
from vision_lab.contracts.schema import parse_run_contract
from vision_lab.metrics import assert_standard_metric_name, direction_for_standard_metric
from vision_lab.task_registry import get_task


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    elif path.suffix.lower() in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    else:
        raise SystemExit(f"Unsupported config extension {path.suffix!r}")
    if not isinstance(data, dict):
        raise SystemExit("Config root must be a JSON object / YAML mapping")
    return data


def _is_run_contract_document(raw: Mapping[str, Any]) -> bool:
    return raw.get("contract_version") == CONTRACT_VERSION and "task" in raw and "backend" in raw


def _promotion_dict(raw: dict[str, Any], *, task_id: str) -> dict[str, Any]:
    spec = get_task(task_id)
    block = raw.get("promotion") if isinstance(raw.get("promotion"), dict) else {}
    primary = str(block.get("primary", spec.primary_metric)).strip()
    assert_standard_metric_name(primary)
    direction_val = direction_for_standard_metric(primary).value
    min_delta = block.get("min_delta", 0.0)
    if not isinstance(min_delta, (int, float)):
        min_delta = 0.0
    secondary = block.get("secondary")
    gates = block.get("gates", [])
    tie_breakers = block.get("tie_breakers", [])
    if not isinstance(gates, list):
        gates = []
    return {
        "primary": primary,
        "direction": direction_val,
        "min_delta": float(min_delta),
        "secondary": secondary if isinstance(secondary, str) or secondary is None else None,
        "gates": gates,
        "tie_breakers": tie_breakers if isinstance(tie_breakers, (list, str)) else [],
    }


def _pipeline_dict(*, task_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "transform_recipe_id": f"legacy.{task_id}",
        "transform_recipe_params": {},
        "collator_id": f"legacy.{task_id}",
        "loss_id": f"legacy.{task_id}",
        "metric_set_id": f"legacy.{task_id}",
        "promotion": _promotion_dict(raw, task_id=task_id),
    }


def _runtime_dict(raw: dict[str, Any]) -> dict[str, Any]:
    seed = int(raw.get("seed", 42))
    if raw.get("bf16"):
        mp = "bf16"
    elif raw.get("fp16"):
        mp = "fp16"
    else:
        mp = "none"
    device = str(raw.get("device", "cuda")).strip() or "cuda"
    workers = int(raw.get("dataloader_num_workers", 4))
    return {
        "seed": seed,
        "mixed_precision": mp,
        "device": device,
        "dataloader_num_workers": workers,
    }


def _backend_for_task(task_id: str) -> str:
    spec = get_task(task_id)
    return "hf_trainer" if spec.backend == "transformers" else "ultralytics"


def _column_mapping_for_task(task_id: str, raw: dict[str, Any]) -> dict[str, str]:
    if task_id == "classify":
        return {
            "image": str(raw.get("image_column_name", "image")),
            "label": str(raw.get("label_column_name", "label")),
        }
    if task_id == "detect":
        return {
            "image": str(raw.get("image_column_name", "image")),
            "objects": str(raw.get("objects_column_name", "objects")),
        }
    if task_id == "segment":
        return {
            "image": str(raw.get("image_column_name", "image")),
            "mask": str(raw.get("mask_column_name", "mask")),
        }
    if task_id == "semantic_segment":
        return {
            "image": str(raw.get("image_column_name", "image")),
            "mask": str(raw.get("mask_column_name", "mask")),
        }
    if task_id in ("instance_segment", "universal_segment"):
        return {
            "image": str(raw.get("image_column_name", "image")),
            "annotation": str(raw.get("annotation_column_name", "annotation")),
        }
    if task_id in ("detect_yolo", "track_yolo", "pose_yolo", "obb_yolo"):
        return {"image": "image", "objects": "objects"}
    if task_id == "segment_yolo":
        return {
            "image": "image",
            "objects": "objects",
            "mask": str(raw.get("mask_column") or "mask"),
        }
    if task_id == "classify_yolo":
        return {
            "image": "image",
            "label": str(raw.get("label_column", "label")),
        }
    raise SystemExit(f"compile_launch_contract: unsupported task {task_id!r}")


def _dataset_revision(raw: dict[str, Any]) -> str:
    rev = raw.get("dataset_revision")
    if rev is None or (isinstance(rev, str) and not rev.strip()):
        return "main"
    return str(rev).strip()


def _hyperparameters_for_contract(task_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    spec = get_task(task_id)
    exclude = {
        "contract_version",
        "task",
        "backend",
        "task_type",
        "dataset_name",
        "dataset_config_name",
        "dataset_revision",
        "dataset_split",
        "train_split",
        "eval_split",
        "image_column_name",
        "label_column_name",
        "mask_column_name",
        "objects_column_name",
        "annotation_column_name",
        "model_name_or_path",
        "model_loader",
        "adaptation_mode",
        "promotion",
        "pipeline",
        "dataset",
        "model",
        "training",
        "runtime",
    }
    if spec.backend == "transformers":
        for k in (
            "model_revision",
            "cache_dir",
            "token",
            "trust_remote_code",
            "config_name",
            "image_processor_name",
            "ignore_mismatched_sizes",
        ):
            exclude.add(k)
    hp = {k: v for k, v in raw.items() if k not in exclude}
    return hp


def _architecture_hints(raw: dict[str, Any], *, task_id: str) -> dict[str, Any]:
    spec = get_task(task_id)
    if spec.backend != "transformers":
        hints: dict[str, Any] = {}
        ocf = raw.get("objects_category_field")
        if ocf is not None:
            hints["objects_category_field"] = ocf
        return hints

    hints = {}
    if raw.get("adaptation_mode") is not None:
        hints["adaptation_mode"] = raw["adaptation_mode"]
    for key in (
        "model_revision",
        "cache_dir",
        "token",
        "trust_remote_code",
        "config_name",
        "image_processor_name",
        "ignore_mismatched_sizes",
    ):
        if raw.get(key) is not None:
            hints[key] = raw[key]
    hints.setdefault("adaptation_mode", "full_finetune")
    hints.setdefault("model_revision", "main")
    hints.setdefault("ignore_mismatched_sizes", True)
    hints.setdefault("trust_remote_code", False)
    return hints


def experiment_config_to_contract_dict(*, task_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Build a run-contract mapping from a legacy flat experiment config dict."""
    spec = get_task(task_id)
    if raw.get("task_type") is not None and str(raw["task_type"]).strip() != task_id:
        raise SystemExit(
            f"task_type mismatch: CLI/task={task_id!r} but config has task_type={raw.get('task_type')!r}"
        )

    dataset_name = raw.get("dataset_name")
    if not dataset_name or not isinstance(dataset_name, str):
        raise SystemExit("dataset_name is required in experiment config")
    split = str(raw.get("dataset_split") or raw.get("train_split") or "train").strip()

    return {
        "contract_version": CONTRACT_VERSION,
        "task": task_id,
        "backend": _backend_for_task(task_id),
        "dataset": {
            "source": "hf_hub",
            "identifier": str(dataset_name).strip(),
            "revision": _dataset_revision(raw),
            "config_name": raw.get("dataset_config_name"),
            "split": split,
            "profile_id": f"{spec.dataset_schema_kind}.legacy",
            "column_mapping": _column_mapping_for_task(task_id, raw),
        },
        "model": {
            "model_id": _model_id_or_exit(raw),
            "loader_strategy": str(raw.get("model_loader", "auto_task_head")).strip(),
            "architecture_hints": _architecture_hints(raw, task_id=task_id),
        },
        "pipeline": _pipeline_dict(task_id=task_id, raw=raw),
        "training": {"hyperparameters": _hyperparameters_for_contract(task_id, raw)},
        "runtime": _runtime_dict(raw),
    }


def _model_id_or_exit(raw: dict[str, Any]) -> str:
    mid = str(raw.get("model_name_or_path", "")).strip()
    if not mid:
        raise SystemExit("model_name_or_path is required in experiment config")
    return mid


def compile_config_file_to_path(*, task_id: str, config_path: Path, output_path: Path) -> None:
    raw = _load_mapping(config_path)
    if _is_run_contract_document(raw):
        contract = parse_run_contract(raw)
        if contract.task != task_id:
            raise SystemExit(
                f"Run contract task {contract.task!r} does not match launch task {task_id!r}"
            )
        payload = run_contract_to_primitive_dict(contract)
    else:
        payload = experiment_config_to_contract_dict(task_id=task_id, raw=raw)
        parse_run_contract(payload)  # validate
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "Usage: python -m vision_lab.compile_launch_contract <task_id> <config.yaml|json> <out.yaml>"
        )
    task_id = sys.argv[1].strip()
    cfg = Path(sys.argv[2]).expanduser().resolve()
    out = Path(sys.argv[3]).expanduser().resolve()
    if not cfg.is_file():
        raise SystemExit(f"Config not found: {cfg}")
    get_task(task_id)  # validate task id
    compile_config_file_to_path(task_id=task_id, config_path=cfg, output_path=out)


if __name__ == "__main__":
    main()
