"""Compile an experiment YAML/JSON config into a validated ``RunContract`` file.

Training entrypoints accept only a resolved contract path. Launchers call this module when
the config is still a **flat** experiment YAML: we run Hub **inspect → profile resolve →
pipeline resolve** for tasks that have resolver wiring.

If resolution cannot run (unsupported task, local dataset path, or Hub/resolver failure),
compilation **aborts** with remediation hints — there is no implicit ``legacy.*`` pipeline
or guessed column mapping (Phase 6).
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn

import yaml

from vision_lab.contracts.run_contract import CONTRACT_VERSION, run_contract_to_primitive_dict
from vision_lab.contracts.schema import parse_run_contract
from vision_lab.metrics import assert_standard_metric_name, direction_for_standard_metric
from vision_lab.resolution.inspect_dataset import inspect_hf_hub_dataset
from vision_lab.resolution.pipeline_resolver import (
    ModelCapabilities,
    PipelineResolutionError,
    resolve_contract_pipeline,
)
from vision_lab.resolution.profile_resolver import (
    ProfileResolutionError,
    resolve_contract_dataset,
)
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


def _dataset_revision(raw: dict[str, Any]) -> str:
    rev = raw.get("dataset_revision")
    if rev is None or (isinstance(rev, str) and not rev.strip()):
        return "main"
    return str(rev).strip()


def _hub_token(raw: dict[str, Any]) -> str | None:
    tok = raw.get("token")
    if isinstance(tok, str) and tok.strip():
        return tok.strip()
    env = os.environ.get("HF_TOKEN") or os.environ.get("hfjob")
    if isinstance(env, str) and env.strip():
        return env.strip()
    return None


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
        "profile_resolver_id",
        "pipeline_spec_id",
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


def _default_profile_and_pipeline(task_id: str) -> tuple[str, str] | None:
    spec = get_task(task_id)
    if task_id == "classify" and spec.backend == "transformers":
        return ("hf_hub.classify.image_label_v1", "hf_trainer.classify.default_v1")
    if task_id == "classify_yolo" and spec.backend == "ultralytics":
        return ("hf_hub.classify.image_label_v1", "ultralytics.yolo.train_v1")
    if task_id == "detect" and spec.backend == "transformers":
        return ("hf_hub.detect.objects_column_v1", "hf_trainer.detect.default_v1")
    if task_id in ("detect_yolo", "track_yolo", "pose_yolo", "obb_yolo") and spec.backend == "ultralytics":
        return ("hf_hub.yolo.detect_labels_role_v1", "ultralytics.yolo.train_v1")
    return None


def _contract_dataset_to_mapping(ds: Any) -> dict[str, Any]:
    return {
        "source": ds.source,
        "identifier": ds.identifier,
        "revision": ds.revision,
        "config_name": ds.config_name,
        "split": ds.split,
        "profile_id": ds.profile_id,
        "column_mapping": dict(ds.column_mapping),
    }


def _pipeline_to_mapping_with_yaml_promotion(
    pipe: Any, raw: dict[str, Any], *, task_id: str
) -> dict[str, Any]:
    return {
        "transform_recipe_id": pipe.transform_recipe_id,
        "transform_recipe_params": dict(pipe.transform_recipe_params),
        "collator_id": pipe.collator_id,
        "loss_id": pipe.loss_id,
        "metric_set_id": pipe.metric_set_id,
        "promotion": _promotion_dict(raw, task_id=task_id),
    }


def _try_hub_resolved_contract_dict(*, task_id: str, raw: dict[str, Any]) -> dict[str, Any] | None:
    """Return a full contract dict when Hub inspect + resolvers apply; else ``None``."""
    defaults = _default_profile_and_pipeline(task_id)
    if defaults is None:
        return None
    default_prof, default_pipe = defaults
    profile_resolver_id = str(raw.get("profile_resolver_id") or default_prof).strip()
    pipeline_spec_id = str(raw.get("pipeline_spec_id") or default_pipe).strip()

    dataset_name = raw.get("dataset_name")
    if not dataset_name or not isinstance(dataset_name, str):
        return None
    if Path(dataset_name).exists():
        return None

    split = str(raw.get("dataset_split") or raw.get("train_split") or "train").strip()
    rev = _dataset_revision(raw)
    token = _hub_token(raw)

    hub = inspect_hf_hub_dataset(
        str(dataset_name).strip(),
        revision=rev if rev != "main" else None,
        config_name=raw.get("dataset_config_name"),
        split=split,
        token=token,
    )
    if hub.hard_errors:
        raise SystemExit(
            "Hub dataset inspection failed:\n  " + "\n  ".join(hub.hard_errors)
        )

    try:
        pr = resolve_contract_dataset(
            task_id=task_id, profile=hub, resolver_id=profile_resolver_id
        )
    except ProfileResolutionError as e:
        raise SystemExit(f"Profile resolution failed ({profile_resolver_id}): {e}") from e

    ds = pr.contract_dataset
    model = ModelCapabilities(
        model_id=_model_id_or_exit(raw),
        loader_strategy=str(raw.get("model_loader", "auto_task_head")).strip(),
        hints=dict(_architecture_hints(raw, task_id=task_id)),
    )
    try:
        pl = resolve_contract_pipeline(
            task_id=task_id,
            dataset=ds,
            model_capabilities=model,
            pipeline_spec_id=pipeline_spec_id,
            hub_profile=hub,
        )
    except PipelineResolutionError as e:
        raise SystemExit(f"Pipeline resolution failed ({pipeline_spec_id}): {e}") from e

    pipe_map = _pipeline_to_mapping_with_yaml_promotion(pl.contract_pipeline, raw, task_id=task_id)

    return {
        "contract_version": CONTRACT_VERSION,
        "task": task_id,
        "backend": _backend_for_task(task_id),
        "dataset": _contract_dataset_to_mapping(ds),
        "model": {
            "model_id": model.model_id,
            "loader_strategy": model.loader_strategy,
            "architecture_hints": dict(model.hints),
        },
        "pipeline": pipe_map,
        "training": {"hyperparameters": _hyperparameters_for_contract(task_id, raw)},
        "runtime": _runtime_dict(raw),
    }


def _raise_flat_config_unresolved(*, task_id: str, raw: dict[str, Any]) -> NoReturn:
    """Fail compile when flat YAML cannot be turned into a contract via Hub resolution."""
    reasons: list[str] = []
    ds = raw.get("dataset_name")
    if isinstance(ds, str) and ds.strip() and Path(ds.strip()).exists():
        reasons.append(
            f"dataset_name looks like a local path ({ds!r}). "
            "Use a Hugging Face Hub dataset id, or pass a complete RunContract YAML as your config."
        )
    if _default_profile_and_pipeline(task_id) is None:
        reasons.append(
            f"task {task_id!r} has no default Hub profile/pipeline wiring in compile_launch_contract; "
            "hand-author a RunContract file (contract_version: 1, task, backend, dataset, model, pipeline, …)."
        )
    if not reasons:
        reasons.append(
            "Hub dataset inspection + profile/pipeline resolution did not produce a contract "
            "(check HF_TOKEN, dataset id, revision, split, and resolver compatibility)."
        )
    raise SystemExit(
        "Cannot compile flat experiment YAML without successful Hub resolution.\n"
        + "\n".join(f"  • {line}" for line in reasons)
        + "\n\nRemedies:\n"
        "  • Fix Hub access and dataset metadata, or\n"
        "  • Store a full RunContract YAML and point --config at it (launchers validate and forward it)."
    )


def experiment_config_to_contract_dict(*, task_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Build a run-contract mapping from a flat experiment config dict (Hub resolution only)."""
    if raw.get("task_type") is not None and str(raw["task_type"]).strip() != task_id:
        raise SystemExit(
            f"task_type mismatch: CLI/task={task_id!r} but config has task_type={raw.get('task_type')!r}"
        )

    resolved = _try_hub_resolved_contract_dict(task_id=task_id, raw=raw)
    if resolved is not None:
        return resolved

    _raise_flat_config_unresolved(task_id=task_id, raw=raw)


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
