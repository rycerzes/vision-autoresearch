"""Type-based dataset column → model input alignment.

No hardcoded mappings. Column matching is driven entirely by:
1. Dataset feature types (from HF datasets)
2. Processor capabilities (what modalities it accepts)
3. Model forward() signature (what kwargs it accepts)
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Feature type classification — uses HF datasets class names directly

def classify_feature(feature: Any) -> str:
    """Return the HF datasets feature type name, normalized to lowercase.

    No mapping table — just uses the actual class name from the datasets library.
    """
    type_name = type(feature).__name__

    # Unwrap Value dtype for finer granularity
    if type_name == "Value":
        dtype_str = str(getattr(feature, "dtype", ""))
        if "float" in dtype_str:
            return "value_float"
        if "int" in dtype_str:
            return "value_int"
        if "string" in dtype_str or "str" in dtype_str:
            return "value_string"
        return "value"

    # Sequence with inner dict → structured (detection objects, etc.)
    if type_name == "Sequence":
        inner = getattr(feature, "feature", None)
        if inner is not None and hasattr(inner, "keys"):
            return "dict"
        return type_name.lower()

    # Dict-like features (bare dict columns like cppe-5's "objects")
    if isinstance(feature, dict) or (hasattr(feature, "keys") and type_name not in ("Image", "ClassLabel", "Audio", "Video")):
        return "dict"

    # Everything else: use the class name directly
    return type_name.lower()  # "Image" → "image", "ClassLabel" → "classlabel", "Audio" → "audio"


# Processor-driven input requirements

def _probe_processor_needs(processor: Any) -> list[str]:
    """Inspect what modalities a processor accepts."""
    needs: list[str] = []

    if hasattr(processor, "image_processor") or type(processor).__name__ in ("AutoImageProcessor", "BaseImageProcessor"):
        needs.append("image")

    if hasattr(processor, "tokenizer"):
        needs.append("text")

    # If it's a bare image processor (not wrapped in AutoProcessor)
    if not needs and hasattr(processor, "preprocess"):
        needs.append("image")

    return needs or ["image"]  # bare minimum assumption: vision models need images


def _probe_model_label_format(model: Any) -> str | None:
    """Inspect model.forward() to determine what label format it expects.

    Returns a feature type string or None if labels aren't accepted.
    """
    forward_fn = getattr(model, "forward", None)
    if forward_fn is None:
        return None

    sig = inspect.signature(forward_fn)
    params = set(sig.parameters.keys())

    # No labels parameter → self-supervised or inference-only
    if "labels" not in params:
        return None

    return "any"  # accepts labels — format determined during probe


# Column mapping

# Feature types that represent image data
_IMAGE_TYPES = frozenset({"image"})
# Feature types that represent text
_TEXT_TYPES = frozenset({"value_string"})
# Feature types that represent class targets
_CLASS_TYPES = frozenset({"classlabel", "value_int"})
# Feature types that represent structured annotations (detection, keypoint)
_STRUCT_TYPES = frozenset({"dict", "sequence"})


def auto_map_columns(
    features: dict[str, Any],
    head_category: str,
    *,
    processor: Any | None = None,
) -> dict[str, Any]:
    """Map dataset columns to model inputs.  Purely type-based.

    Uses processor capabilities when available, otherwise infers from
    head_category what the model needs.

    Returns ``{"image": "col_name", "target": "col_name", ...}``.
    Raises ``ValueError`` with a clear message when mapping fails.
    """
    # Classify all columns by type
    col_types: dict[str, str] = {}
    for col_name, feat in features.items():
        col_types[col_name] = classify_feature(feat)

    logger.info("Column types: %s", col_types)

    # Find what the model needs from the processor
    if processor is not None:
        proc_needs = _probe_processor_needs(processor)
    else:
        proc_needs = ["image"]  # vision models always need images

    mapping: dict[str, Any] = {}

    image_cols = [c for c, t in col_types.items() if t in _IMAGE_TYPES]

    if "image" in proc_needs:
        if not image_cols:
            raise ValueError(
                "Model requires image input but no Image-type column found. "
                "Dataset columns: %s. Set `column_map` in YAML." % col_types
            )
        mapping["image"] = image_cols[0]

    if "text" in proc_needs:
        text_cols = [c for c, t in col_types.items() if t in _TEXT_TYPES and c not in mapping.values()]
        if text_cols:
            mapping["text"] = text_cols[0]

    used = set(mapping.values())
    target = _find_target(col_types, head_category, used)
    if target is not None:
        mapping["target"] = target

    # For detection-like targets, also map sub-fields within the structured dict
    if (
        "target" in mapping
        and head_category in ("detection", "structured_detection")
    ):
        target_col = mapping["target"]
        target_feat = features.get(target_col)
        if target_feat is not None:
            sub_map = auto_map_target_subfields(target_feat)
            if sub_map:
                mapping["target_subfields"] = sub_map

    logger.info("Auto-mapped columns for %s: %s", head_category, mapping)
    return mapping


def _find_target(
    col_types: dict[str, str],
    head_category: str,
    used: set[str],
) -> str | None:
    """Find the target/label column based on what's available."""

    # Self-supervised: no target needed
    if "self_supervised" in head_category or "masked" in head_category:
        return None

    unused = {c: t for c, t in col_types.items() if c not in used}

    # Classification-like: look for ClassLabel or int column
    if "classif" in head_category or "contrastive" in head_category:
        for col, typ in unused.items():
            if typ in _CLASS_TYPES:
                return col
        # Contrastive may use text as target
        for col, typ in unused.items():
            if typ in _TEXT_TYPES:
                return col

    # Detection-like: look for structured dict
    if "detect" in head_category:
        for col, typ in unused.items():
            if typ in _STRUCT_TYPES:
                return col

    # Dense prediction (segmentation, depth): look for second Image column
    if "dense" in head_category or "segment" in head_category or "depth" in head_category or "prompted" in head_category:
        for col, typ in unused.items():
            if typ in _IMAGE_TYPES:
                return col

    # Sequence generation (OCR, table): look for text
    if "sequence" in head_category or "generation" in head_category:
        for col, typ in unused.items():
            if typ in _TEXT_TYPES:
                return col

    # Image reconstruction: look for second Image column
    if "reconstruct" in head_category or "image_to_image" in head_category:
        for col, typ in unused.items():
            if typ in _IMAGE_TYPES:
                return col

    # Pair matching: look for second Image column
    if "match" in head_category or "pair" in head_category:
        for col, typ in unused.items():
            if typ in _IMAGE_TYPES:
                return col

    # Fallback: first unused column of any type that isn't a simple value
    for col, typ in unused.items():
        if typ not in ("value_int", "value_float", "value"):
            return col

    return None


def auto_map_target_subfields(feature: Any) -> dict[str, str] | None:
    """Type-based sub-field mapping within a structured detection target.

    Scans the HF feature schema of a structured dict column to identify
    which sub-fields are bboxes, category labels, areas, etc.  Uses
    *only* feature types — never field names.

    Handles two common layouts:
    - ``Sequence({"bbox": Sequence(float, 4), "category": ClassLabel, ...})``
    - ``{"bbox": List(List(float, 4)), "category": List(ClassLabel), ...}``  (cppe-5 style)

    Returns ``{"bbox": "actual_field", "category": "actual_field", ...}``
    or ``None`` if no sub-fields can be identified.
    """
    # Get sub-field features from the structured target
    sub_features = _unwrap_to_subfields(feature)
    if sub_features is None or not sub_features:
        return None

    result: dict[str, str] = {}
    first_int_field: str | None = None

    for field_name, feat in sub_features.items():
        # Unwrap one level of List/Sequence to get the per-object feature type
        inner_feat = _unwrap_list(feat)

        inner_type = classify_feature(inner_feat)

        # ClassLabel → category
        if inner_type == "classlabel" and "category" not in result:
            result["category"] = field_name
            continue

        # Sequence/List of exactly 4 floats → bounding box
        if _is_bbox_feature(inner_feat) and "bbox" not in result:
            result["bbox"] = field_name
            continue

        # Variable-length float sequence → polygon/segmentation
        if _is_polygon_feature(inner_feat) and "segmentation" not in result:
            result["segmentation"] = field_name
            continue

        # Track first int field as potential category fallback
        if inner_type in ("value_int", "value") and first_int_field is None:
            first_int_field = field_name

    # If no ClassLabel found, use the first integer as category
    if "category" not in result and first_int_field is not None:
        result["category"] = first_int_field

    return result if result else None


def _unwrap_to_subfields(feature: Any) -> dict[str, Any] | None:
    """Extract sub-field features from a structured target column.

    Handles bare dicts, Sequence(dict), and dict-of-Lists.
    """
    # Already a dict of features
    if isinstance(feature, dict):
        return dict(feature)

    # Sequence(dict)
    if hasattr(feature, "feature"):
        inner = feature.feature
        if isinstance(inner, dict):
            return dict(inner)
        if hasattr(inner, "keys"):
            try:
                return {k: inner[k] for k in inner.keys()}
            except Exception:
                pass

    # Has .keys() (dict-like HF feature)
    if hasattr(feature, "keys") and callable(getattr(feature, "keys")):
        try:
            keys = list(feature.keys())
            return {k: feature[k] for k in keys}
        except Exception:
            pass

    return None


def _unwrap_list(feature: Any) -> Any:
    """Unwrap one level of List/Sequence to get the per-item feature type.

    ``List(ClassLabel)`` → ``ClassLabel``
    ``List(List(float, 4))`` → ``List(float, 4)``  (which is a bbox)
    ``Sequence(float, 4)`` → stays as-is (already a bbox feature)
    """
    type_name = type(feature).__name__
    # HF datasets uses both "Sequence" and "LargeList" / "List" class
    if type_name in ("Sequence", "LargeList", "List"):
        inner = getattr(feature, "feature", None)
        if inner is not None:
            return inner
    return feature


def _is_bbox_feature(feature: Any) -> bool:
    """Check if a feature looks like a bounding box: Sequence/List of 4 floats."""
    type_name = type(feature).__name__
    if type_name not in ("Sequence", "List", "LargeList"):
        return False
    length = getattr(feature, "length", -1)
    if length != 4:
        return False
    inner_feat = getattr(feature, "feature", None)
    if inner_feat is None:
        return False
    inner_type = classify_feature(inner_feat)
    return inner_type in ("value_float", "value_int", "value")


def _is_polygon_feature(feature: Any) -> bool:
    """Check if a feature looks like a polygon: Sequence/List of floats (variable length)."""
    type_name = type(feature).__name__
    if type_name not in ("Sequence", "List", "LargeList"):
        return False
    length = getattr(feature, "length", -1)
    if length == 4:  # that's a bbox, not a polygon
        return False
    inner_feat = getattr(feature, "feature", None)
    if inner_feat is None:
        return False
    # Could be nested Sequence (list of polygons)
    inner_type = classify_feature(inner_feat)
    return inner_type in ("value_float", "sequence")
