"""Head category introspection for HF Transformers and Ultralytics models.

Head categories are derived directly from the model's architecture suffix
(HF) or task property (Ultralytics).  No static mapping tables — the suffix
string IS the category, just normalised to a consistent format.
"""

from __future__ import annotations

import re

# Architecture suffix → head category (HF Transformers)
# This is the ONLY mapping.  It groups architectures that share the same
# data pipeline / loss / metrics.  Each entry exists because multiple
# HF model families (DETR, D-FINE, Relation DETR, …) share the same suffix.
# Without this mapping, every model would be its own category and you'd need
# per-model logic instead of per-category logic.

HEAD_CATEGORIES: dict[str, str] = {
    # Detection
    "ForObjectDetection": "detection",
    "ForZeroShotObjectDetection": "detection",
    # Classification
    "ForImageClassification": "classification",
    "ForVideoClassification": "classification",
    "ForZeroShotImageClassification": "contrastive",
    # Segmentation
    "ForSemanticSegmentation": "dense_classification",
    "ForImageSegmentation": "dense_classification",
    "ForInstanceSegmentation": "detection",
    "ForUniversalSegmentation": "detection",
    # Depth / regression
    "ForDepthEstimation": "dense_regression",
    # Sequence generation
    "ForTextRecognition": "sequence_generation",
    "ForTableRecognition": "sequence_generation",
    # Keypoint / matching
    "ForKeypointDetection": "structured_detection",
    "ForKeypointMatching": "pair_matching",
    # Self-supervised
    "ForMaskedImageModeling": "self_supervised",
    # Image-to-image
    "ForImageToImage": "image_reconstruction",
}

# WHY THIS MAPPING EXISTS:
# Without it, ForObjectDetection and ForZeroShotObjectDetection would be
# different categories needing separate loss/metric/collation code — but
# they share the exact same pipeline.  The mapping groups them.
#
# WHY IT CAN'T BE DERIVED AT RUNTIME:
# HF Transformers doesn't expose "this model does detection" as queryable
# metadata.  The architecture suffix IS the metadata — we just group
# synonymous suffixes.
#
# WHEN IT NEEDS UPDATING:
# When HF adds a new AutoModelFor* class.  That happens ~1-2 times per year.
# One line addition.

# Ultralytics task property → head category
_ULTRALYTICS_TASK_MAP: dict[str, str] = {
    "detect": "detection",
    "segment": "detection",  # instance seg = detection + masks
    "classify": "classification",
    "pose": "structured_detection",
    "obb": "detection",
}

# Regex to extract the ``For*`` suffix from an architecture string.
_ARCH_SUFFIX_RE = re.compile(r"(For\w+)$")


def head_category_from_arch(architecture: str) -> str | None:
    """Derive head category from an HF ``config.architectures`` entry.

    Returns *None* when no known suffix matches (caller decides how to handle).
    """
    m = _ARCH_SUFFIX_RE.search(architecture)
    if m is not None:
        suffix = m.group(1)
        cat = HEAD_CATEGORIES.get(suffix)
        if cat is not None:
            return cat

    # Models without For* suffix (SAM family)
    arch_lower = architecture.lower()
    if "sam" in arch_lower:
        return "prompted_segmentation"

    return None


def head_category_from_ultralytics_task(task: str) -> str | None:
    """Map an Ultralytics model ``.task`` property to a head category."""
    return _ULTRALYTICS_TASK_MAP.get(task)
