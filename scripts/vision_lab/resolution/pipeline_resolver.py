"""Resolve training pipeline fields (transform, collator, loss, metrics) before execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from vision_lab.contracts.run_contract import (
    ContractDataset,
    ContractPipeline,
    ContractPromotion,
    JSONValue,
)
from vision_lab.metrics import direction_for_standard_metric
from vision_lab.resolution.inspect_dataset import HubDatasetProfile
from vision_lab.task_registry import get_task, promotion_metric_for_task


@dataclass(frozen=True)
class ModelCapabilities:
    """Model identity and hints supplied by the compile step (no silent inference)."""

    model_id: str
    loader_strategy: str
    hints: dict[str, JSONValue]


@dataclass(frozen=True)
class PipelineResolution:
    """Outcome of applying one registered pipeline spec."""

    pipeline_spec_id: str
    contract_pipeline: ContractPipeline
    confidence: float
    rationale: str


class PipelineResolutionError(ValueError):
    """Pipeline spec cannot be applied to the given task, dataset contract, or model."""


PipelineResolverFn = Callable[
    [str, ContractDataset, ModelCapabilities, HubDatasetProfile | None],
    ContractPipeline,
]

_PIPELINE_RESOLVERS: dict[str, PipelineResolverFn] = {}


def register_pipeline_resolver(pipeline_spec_id: str, fn: PipelineResolverFn) -> None:
    if not pipeline_spec_id or not pipeline_spec_id.strip():
        raise ValueError("pipeline_spec_id must be non-empty")
    _PIPELINE_RESOLVERS[pipeline_spec_id] = fn


def list_pipeline_resolver_ids() -> tuple[str, ...]:
    return tuple(sorted(_PIPELINE_RESOLVERS))


def resolve_contract_pipeline(
    *,
    task_id: str,
    dataset: ContractDataset,
    model_capabilities: ModelCapabilities,
    pipeline_spec_id: str,
    hub_profile: HubDatasetProfile | None = None,
) -> PipelineResolution:
    """
    Produce a ``ContractPipeline`` using exactly ``pipeline_spec_id``.

    Column wiring comes only from ``dataset.column_mapping`` (canonical role -> column).
    When ``hub_profile`` is set, each mapped column name must appear in the profile's
    ``feature_schema`` so compile-time wiring matches hub features.
    """
    fn = _PIPELINE_RESOLVERS.get(pipeline_spec_id)
    if fn is None:
        raise PipelineResolutionError(
            f"unknown pipeline_spec_id {pipeline_spec_id!r} "
            f"(known: {', '.join(list_pipeline_resolver_ids())})"
        )
    if hub_profile is not None and hub_profile.hard_errors:
        raise PipelineResolutionError(
            "hub_profile has inspection errors: " + "; ".join(hub_profile.hard_errors)
        )
    if hub_profile is not None:
        _assert_hub_column_alignment(dataset, hub_profile)
    try:
        pipe = fn(task_id, dataset, model_capabilities, hub_profile)
    except PipelineResolutionError:
        raise
    except Exception as e:
        raise PipelineResolutionError(str(e)) from e
    rationale = (
        f"pipeline_spec={pipeline_spec_id!r}; task={task_id!r}; "
        f"transform={pipe.transform_recipe_id!r}; collator={pipe.collator_id!r}; "
        f"loss={pipe.loss_id!r}; metrics={pipe.metric_set_id!r}"
    )
    return PipelineResolution(
        pipeline_spec_id=pipeline_spec_id,
        contract_pipeline=pipe,
        confidence=1.0,
        rationale=rationale,
    )


def _roles(dataset: ContractDataset) -> dict[str, str]:
    return dict(dataset.column_mapping)


def _require_roles(dataset: ContractDataset, *roles: str) -> dict[str, str]:
    m = _roles(dataset)
    missing = [r for r in roles if r not in m]
    if missing:
        raise PipelineResolutionError(
            f"column_mapping missing required role(s) {missing!r}; have roles {sorted(m)}"
        )
    return m


def _assert_hub_column_alignment(dataset: ContractDataset, profile: HubDatasetProfile) -> None:
    declared = frozenset(col for _, col in dataset.column_mapping)
    hub_cols = frozenset(name for name, _ in profile.feature_schema)
    unknown = sorted(declared - hub_cols)
    if unknown:
        raise PipelineResolutionError(
            f"column_mapping references columns not present in hub feature_schema: {unknown}; "
            f"hub columns: {sorted(hub_cols)}"
        )


def _default_contract_promotion(task_id: str) -> ContractPromotion:
    primary = promotion_metric_for_task(task_id)
    direction = direction_for_standard_metric(primary)
    return ContractPromotion(
        primary=primary,
        direction=direction,
        min_delta=0.0,
        secondary=None,
        gates=(),
        tie_breakers=(),
    )


def _assert_task_schema(task_id: str, expected: str) -> None:
    spec = get_task(task_id)
    if spec.dataset_schema_kind != expected:
        raise PipelineResolutionError(
            f"task {task_id!r} has dataset_schema_kind={spec.dataset_schema_kind!r}, "
            f"expected {expected!r} for this pipeline spec"
        )


def _assert_task_backend(task_id: str, expected: str) -> None:
    spec = get_task(task_id)
    if spec.backend != expected:
        raise PipelineResolutionError(
            f"task {task_id!r} uses backend {spec.backend!r}, expected {expected!r}"
        )


def _transform_params_from_hints(hints: dict[str, JSONValue]) -> dict[str, JSONValue]:
    allowed = frozenset(
        {
            "image_size",
            "image_square_size",
            "use_albumentations",
            "use_trivial_augment",
            "adaptation_mode",
        }
    )
    return {k: v for k, v in hints.items() if k in allowed}


def _resolve_hf_classify_default_v1(
    task_id: str,
    dataset: ContractDataset,
    model: ModelCapabilities,
    _hub: HubDatasetProfile | None,
) -> ContractPipeline:
    _assert_task_backend(task_id, "transformers")
    _assert_task_schema(task_id, "classification")
    _require_roles(dataset, "image", "label")
    params = _transform_params_from_hints(model.hints)
    return ContractPipeline(
        transform_recipe_id="hf_image_cls_default",
        transform_recipe_params=params,
        collator_id="hf_pixel_values_classification",
        loss_id="cross_entropy",
        metric_set_id="hf.classify.accuracy",
        promotion=_default_contract_promotion(task_id),
    )


def _resolve_hf_detect_default_v1(
    task_id: str,
    dataset: ContractDataset,
    model: ModelCapabilities,
    _hub: HubDatasetProfile | None,
) -> ContractPipeline:
    _assert_task_backend(task_id, "transformers")
    _assert_task_schema(task_id, "detection")
    _require_roles(dataset, "image", "objects")
    params = _transform_params_from_hints(model.hints)
    return ContractPipeline(
        transform_recipe_id="hf_detection_coco_default",
        transform_recipe_params=params,
        collator_id="hf_detection_batch",
        loss_id="detection_loss",
        metric_set_id="hf.detect.map",
        promotion=_default_contract_promotion(task_id),
    )


def _resolve_hf_segment_semantic_mask_v1(
    task_id: str,
    dataset: ContractDataset,
    model: ModelCapabilities,
    _hub: HubDatasetProfile | None,
) -> ContractPipeline:
    _assert_task_backend(task_id, "transformers")
    _assert_task_schema(task_id, "semantic_segmentation")
    _require_roles(dataset, "image", "mask")
    params = _transform_params_from_hints(model.hints)
    return ContractPipeline(
        transform_recipe_id="hf_semantic_mask_default",
        transform_recipe_params=params,
        collator_id="hf_semantic_mask_batch",
        loss_id="semantic_segmentation_loss",
        metric_set_id="hf.segment.miou",
        promotion=_default_contract_promotion(task_id),
    )


def _resolve_hf_instance_objects_v1(
    task_id: str,
    dataset: ContractDataset,
    model: ModelCapabilities,
    _hub: HubDatasetProfile | None,
) -> ContractPipeline:
    _assert_task_backend(task_id, "transformers")
    _assert_task_schema(task_id, "instance_segmentation")
    _require_roles(dataset, "image", "objects")
    params = _transform_params_from_hints(model.hints)
    return ContractPipeline(
        transform_recipe_id="hf_instance_mask_default",
        transform_recipe_params=params,
        collator_id="hf_instance_batch",
        loss_id="instance_segmentation_loss",
        metric_set_id="hf.instance.mask_map",
        promotion=_default_contract_promotion(task_id),
    )


def _resolve_hf_panoptic_mask_segments_v1(
    task_id: str,
    dataset: ContractDataset,
    model: ModelCapabilities,
    _hub: HubDatasetProfile | None,
) -> ContractPipeline:
    _assert_task_backend(task_id, "transformers")
    _assert_task_schema(task_id, "panoptic_segmentation")
    _require_roles(dataset, "image", "mask", "segments")
    params = _transform_params_from_hints(model.hints)
    return ContractPipeline(
        transform_recipe_id="hf_panoptic_default",
        transform_recipe_params=params,
        collator_id="hf_panoptic_batch",
        loss_id="panoptic_segmentation_loss",
        metric_set_id="hf.panoptic.pq",
        promotion=_default_contract_promotion(task_id),
    )


def _resolve_ultralytics_yolo_train_v1(
    task_id: str,
    dataset: ContractDataset,
    model: ModelCapabilities,
    _hub: HubDatasetProfile | None,
) -> ContractPipeline:
    _assert_task_backend(task_id, "ultralytics")
    spec = get_task(task_id)
    kind = spec.dataset_schema_kind
    params = _transform_params_from_hints(model.hints)
    if kind == "classification":
        _require_roles(dataset, "image", "label")
        return ContractPipeline(
            transform_recipe_id="ultralytics_cls_default",
            transform_recipe_params=params,
            collator_id="ultralytics_cls_batch",
            loss_id="ultralytics_cls_loss",
            metric_set_id="ultralytics.classify.accuracy",
            promotion=_default_contract_promotion(task_id),
        )
    if kind == "detection":
        _require_roles(dataset, "image", "labels")
        return ContractPipeline(
            transform_recipe_id="ultralytics_detect_default",
            transform_recipe_params=params,
            collator_id="ultralytics_yolo_batch",
            loss_id="ultralytics_detection_loss",
            metric_set_id="ultralytics.detect.map",
            promotion=_default_contract_promotion(task_id),
        )
    if kind == "instance_segmentation":
        _require_roles(dataset, "image", "labels")
        return ContractPipeline(
            transform_recipe_id="ultralytics_segment_default",
            transform_recipe_params=params,
            collator_id="ultralytics_yolo_batch",
            loss_id="ultralytics_segmentation_loss",
            metric_set_id="ultralytics.segment.mask_map",
            promotion=_default_contract_promotion(task_id),
        )
    raise PipelineResolutionError(
        f"ultralytics.yolo.train_v1 does not cover dataset_schema_kind={kind!r} for task {task_id!r}"
    )


def _register_builtin_pipeline_resolvers() -> None:
    register_pipeline_resolver("hf_trainer.classify.default_v1", _resolve_hf_classify_default_v1)
    register_pipeline_resolver("hf_trainer.detect.default_v1", _resolve_hf_detect_default_v1)
    register_pipeline_resolver(
        "hf_trainer.segment.semantic_mask_v1", _resolve_hf_segment_semantic_mask_v1
    )
    register_pipeline_resolver("hf_trainer.instance.objects_v1", _resolve_hf_instance_objects_v1)
    register_pipeline_resolver(
        "hf_trainer.panoptic.mask_segments_v1", _resolve_hf_panoptic_mask_segments_v1
    )
    register_pipeline_resolver("ultralytics.yolo.train_v1", _resolve_ultralytics_yolo_train_v1)


_register_builtin_pipeline_resolvers()
