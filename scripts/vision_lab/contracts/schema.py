"""Strict parsing and validation for ``RunContract``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from vision_lab.contracts.run_contract import (
    CONTRACT_VERSION,
    BackendId,
    ContractDataset,
    ContractGate,
    ContractModel,
    ContractPipeline,
    ContractPromotion,
    ContractRuntime,
    ContractTraining,
    JSONValue,
    MixedPrecisionKind,
    RunContract,
)
from vision_lab.metrics import (
    MetricDirection,
    assert_standard_metric_name,
    direction_for_standard_metric,
)
from vision_lab.task_registry import get_task


class RunContractValidationError(ValueError):
    """Contract JSON/YAML failed schema or semantic checks."""

    def __init__(self, message: str, *, path: str = "") -> None:
        self.path = path
        full = f"{path}: {message}" if path else message
        super().__init__(full)


def _p(parent: str, key: str) -> str:
    return f"{parent}.{key}" if parent else key


def _require_mapping(raw: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise RunContractValidationError(f"expected object, got {type(raw).__name__}", path=path)
    return raw


def _require_str(raw: Any, *, path: str) -> str:
    if not isinstance(raw, str) or raw == "":
        raise RunContractValidationError("expected non-empty string", path=path)
    return raw


def _optional_str(raw: Any, *, path: str) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise RunContractValidationError("expected string or null", path=path)
    return raw or None


def _require_int(raw: Any, *, path: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise RunContractValidationError("expected integer", path=path)
    return raw


def _require_float(raw: Any, *, path: str) -> float:
    if isinstance(raw, bool):
        raise RunContractValidationError("expected number", path=path)
    if isinstance(raw, int):
        return float(raw)
    if isinstance(raw, float):
        return raw
    raise RunContractValidationError("expected number", path=path)


def _optional_float(raw: Any, *, path: str) -> float | None:
    if raw is None:
        return None
    return _require_float(raw, path=path)


def _is_json_value(raw: Any, *, path: str) -> JSONValue:
    """Validate ``raw`` is JSON-serializable primitive structure (no NaN)."""
    if raw is None or isinstance(raw, (str, bool)):
        return raw
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        if raw != raw:  # NaN
            raise RunContractValidationError("NaN is not allowed in contract JSON", path=path)
        return raw
    if isinstance(raw, list):
        return [_is_json_value(x, path=_p(path, f"[{i}]")) for i, x in enumerate(raw)]
    if isinstance(raw, dict):
        out: dict[str, JSONValue] = {}
        for k, v in raw.items():
            if not isinstance(k, str):
                raise RunContractValidationError("object keys must be strings", path=path)
            out[k] = _is_json_value(v, path=_p(path, k))
        return out
    raise RunContractValidationError(
        f"unsupported JSON type {type(raw).__name__}", path=path
    )


def _parse_json_dict(raw: Any, *, path: str) -> dict[str, JSONValue]:
    val = _is_json_value(raw, path=path)
    if not isinstance(val, dict):
        raise RunContractValidationError("expected object", path=path)
    return val


def _normalize_column_mapping(raw: Any, *, path: str) -> tuple[tuple[str, str], ...]:
    """Return (role, column) pairs sorted by role for stable fingerprints."""
    if isinstance(raw, Mapping):
        pairs = [(str(k), str(v)) for k, v in raw.items()]
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        pairs = []
        for i, item in enumerate(raw):
            ip = _p(path, str(i))
            if not isinstance(item, (Sequence, Mapping)) or isinstance(item, (str, bytes)):
                raise RunContractValidationError("expected [role, column] pair or object", path=ip)
            if isinstance(item, Mapping):
                m = _require_mapping(item, path=ip)
                role = _require_str(m.get("role"), path=_p(ip, "role"))
                col = _require_str(m.get("column"), path=_p(ip, "column"))
            else:
                seq = list(item)
                if len(seq) != 2:
                    raise RunContractValidationError("pair must have length 2", path=ip)
                role = _require_str(seq[0], path=_p(ip, "0"))
                col = _require_str(seq[1], path=_p(ip, "1"))
            pairs.append((role, col))
    else:
        raise RunContractValidationError("expected object or array of pairs", path=path)
    if not pairs:
        raise RunContractValidationError("column_mapping must be non-empty", path=path)
    roles = [p[0] for p in pairs]
    if len(set(roles)) != len(roles):
        raise RunContractValidationError("duplicate role in column_mapping", path=path)
    return tuple(sorted(pairs, key=lambda x: x[0]))


def _parse_backend(raw: Any, *, path: str) -> BackendId:
    s = _require_str(raw, path=path)
    if s not in ("hf_trainer", "ultralytics"):
        raise RunContractValidationError(
            "expected 'hf_trainer' or 'ultralytics'", path=path
        )
    return s  # type: ignore[return-value]


def _parse_mixed_precision(raw: Any, *, path: str) -> MixedPrecisionKind:
    s = _require_str(raw, path=path)
    if s not in ("fp16", "bf16", "none"):
        raise RunContractValidationError("expected 'fp16', 'bf16', or 'none'", path=path)
    return s  # type: ignore[return-value]


def _parse_dataset_source(raw: Any, *, path: str) -> str:
    s = _require_str(raw, path=path)
    if s not in ("hf_hub", "local"):
        raise RunContractValidationError("expected 'hf_hub' or 'local'", path=path)
    return s


def _parse_direction(raw: Any, *, path: str) -> MetricDirection:
    s = _require_str(raw, path=path).lower()
    if s == "higher":
        return MetricDirection.HIGHER
    if s == "lower":
        return MetricDirection.LOWER
    raise RunContractValidationError("expected 'higher' or 'lower'", path=path)


def _parse_gate(raw: Any, *, path: str, task_id: str) -> ContractGate:
    m = _require_mapping(raw, path=path)
    metric = _require_str(m.get("metric"), path=_p(path, "metric"))
    assert_standard_metric_name(metric)
    spec = get_task(task_id)
    if metric not in spec.allowed_gate_metrics:
        raise RunContractValidationError(
            f"gate metric {metric!r} not allowed for task {task_id!r}",
            path=_p(path, "metric"),
        )
    gmin = m.get("gate_min", m.get("min"))
    gmax = m.get("gate_max", m.get("max"))
    return ContractGate(
        metric=metric,
        gate_min=_optional_float(gmin, path=_p(path, "min")),
        gate_max=_optional_float(gmax, path=_p(path, "max")),
    )


def _parse_promotion(raw: Any, *, path: str, task_id: str) -> ContractPromotion:
    m = _require_mapping(raw, path=path)
    primary = _require_str(m.get("primary"), path=_p(path, "primary"))
    assert_standard_metric_name(primary)
    spec = get_task(task_id)
    if primary not in spec.allowed_primary_metrics:
        raise RunContractValidationError(
            f"primary metric {primary!r} not allowed for task {task_id!r}",
            path=_p(path, "primary"),
        )
    direction = _parse_direction(m.get("direction"), path=_p(path, "direction"))
    expected = direction_for_standard_metric(primary)
    if direction != expected:
        raise RunContractValidationError(
            f"direction {direction.value!r} does not match standard metric {primary!r} "
            f"(expected {expected.value!r})",
            path=_p(path, "direction"),
        )
    min_delta = _require_float(m.get("min_delta", 0.0), path=_p(path, "min_delta"))
    sec_raw = m.get("secondary")
    secondary: str | None
    if sec_raw in (None, ""):
        secondary = None
    else:
        secondary = _require_str(sec_raw, path=_p(path, "secondary"))
        assert_standard_metric_name(secondary)
        if secondary not in spec.allowed_secondary_metrics:
            raise RunContractValidationError(
                f"secondary metric {secondary!r} not allowed for task {task_id!r}",
                path=_p(path, "secondary"),
            )
    gates_raw = m.get("gates", [])
    if not isinstance(gates_raw, list):
        raise RunContractValidationError("gates must be an array", path=_p(path, "gates"))
    gates = tuple(
        _parse_gate(g, path=_p(path, f"gates[{i}]"), task_id=task_id)
        for i, g in enumerate(gates_raw)
    )
    tb_raw = m.get("tie_breakers", [])
    if tb_raw in (None, ""):
        tie_breakers: tuple[str, ...] = ()
    elif isinstance(tb_raw, str):
        tie_breakers = (_require_str(tb_raw, path=_p(path, "tie_breakers")),)
    elif isinstance(tb_raw, list):
        tie_breakers = tuple(
            _require_str(x, path=_p(path, f"tie_breakers[{i}]")) for i, x in enumerate(tb_raw)
        )
    else:
        raise RunContractValidationError("tie_breakers must be array or string", path=path)
    for i, name in enumerate(tie_breakers):
        assert_standard_metric_name(name)
        if name not in spec.allowed_tie_breaker_metrics:
            raise RunContractValidationError(
                f"tie_breaker metric {name!r} not allowed for task {task_id!r}",
                path=_p(path, f"tie_breakers[{i}]"),
            )
    return ContractPromotion(
        primary=primary,
        direction=direction,
        min_delta=min_delta,
        secondary=secondary,
        gates=gates,
        tie_breakers=tie_breakers,
    )


def _parse_pipeline(raw: Any, *, path: str, task_id: str) -> ContractPipeline:
    m = _require_mapping(raw, path=path)
    return ContractPipeline(
        transform_recipe_id=_require_str(
            m.get("transform_recipe_id"), path=_p(path, "transform_recipe_id")
        ),
        transform_recipe_params=_parse_json_dict(
            m.get("transform_recipe_params", {}), path=_p(path, "transform_recipe_params")
        ),
        collator_id=_require_str(m.get("collator_id"), path=_p(path, "collator_id")),
        loss_id=_require_str(m.get("loss_id"), path=_p(path, "loss_id")),
        metric_set_id=_require_str(m.get("metric_set_id"), path=_p(path, "metric_set_id")),
        promotion=_parse_promotion(m.get("promotion", {}), path=_p(path, "promotion"), task_id=task_id),
    )


def _assert_backend_matches_task(backend: BackendId, task_id: str) -> None:
    spec = get_task(task_id)
    if backend == "hf_trainer" and spec.backend != "transformers":
        raise RunContractValidationError(
            f"backend 'hf_trainer' incompatible with task {task_id!r} (expects ultralytics)",
            path="backend",
        )
    if backend == "ultralytics" and spec.backend != "ultralytics":
        raise RunContractValidationError(
            f"backend 'ultralytics' incompatible with task {task_id!r} (expects hf_trainer)",
            path="backend",
        )


def parse_run_contract(raw: Mapping[str, Any]) -> RunContract:
    """Parse a mapping into ``RunContract``, applying strict validation."""
    root = _require_mapping(raw, path="")
    ver = root.get("contract_version")
    if ver != CONTRACT_VERSION:
        raise RunContractValidationError(
            f"contract_version must be {CONTRACT_VERSION}, got {ver!r}",
            path="contract_version",
        )
    task_id = _require_str(root.get("task"), path="task")
    try:
        get_task(task_id)
    except KeyError as e:
        raise RunContractValidationError(str(e), path="task") from e
    backend = _parse_backend(root.get("backend"), path="backend")
    _assert_backend_matches_task(backend, task_id)

    dm = _require_mapping(root.get("dataset"), path="dataset")
    source = _parse_dataset_source(dm.get("source"), path="dataset.source")
    identifier = _require_str(dm.get("identifier"), path="dataset.identifier")
    revision = _optional_str(dm.get("revision"), path="dataset.revision")
    config_name = _optional_str(dm.get("config_name"), path="dataset.config_name")
    split = _require_str(dm.get("split"), path="dataset.split")
    profile_id = _require_str(dm.get("profile_id"), path="dataset.profile_id")
    column_mapping = _normalize_column_mapping(dm.get("column_mapping"), path="dataset.column_mapping")
    if source == "hf_hub" and revision is None:
        raise RunContractValidationError(
            "revision is required when source is 'hf_hub' (use explicit commit sha or tag)",
            path="dataset.revision",
        )

    mm = _require_mapping(root.get("model"), path="model")
    model = ContractModel(
        model_id=_require_str(mm.get("model_id"), path="model.model_id"),
        loader_strategy=_require_str(mm.get("loader_strategy"), path="model.loader_strategy"),
        architecture_hints=_parse_json_dict(mm.get("architecture_hints", {}), path="model.architecture_hints"),
    )

    pipeline = _parse_pipeline(root.get("pipeline"), path="pipeline", task_id=task_id)

    tm = _require_mapping(root.get("training"), path="training")
    training = ContractTraining(
        hyperparameters=_parse_json_dict(tm.get("hyperparameters", {}), path="training.hyperparameters"),
    )

    rm = _require_mapping(root.get("runtime"), path="runtime")
    runtime = ContractRuntime(
        seed=_require_int(rm.get("seed"), path="runtime.seed"),
        mixed_precision=_parse_mixed_precision(rm.get("mixed_precision"), path="runtime.mixed_precision"),
        device=_require_str(rm.get("device"), path="runtime.device"),
        dataloader_num_workers=_require_int(
            rm.get("dataloader_num_workers"), path="runtime.dataloader_num_workers"
        ),
    )
    if runtime.dataloader_num_workers < 0:
        raise RunContractValidationError("dataloader_num_workers must be >= 0", path="runtime.dataloader_num_workers")

    return RunContract(
        contract_version=CONTRACT_VERSION,
        task=task_id,
        backend=backend,
        dataset=ContractDataset(
            source=source,  # type: ignore[arg-type]
            identifier=identifier,
            revision=revision,
            config_name=config_name,
            split=split,
            profile_id=profile_id,
            column_mapping=column_mapping,
        ),
        model=model,
        pipeline=pipeline,
        training=training,
        runtime=runtime,
    )
