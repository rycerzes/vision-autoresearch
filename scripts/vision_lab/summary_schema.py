"""Structured summary contract for the ``VISION AUTORESEARCH SUMMARY`` log block.

Trainers emit ``key: value`` lines between markers. Standard headline metrics
(see ``vision_lab.metrics.STANDARD_METRICS``) are task-scoped: only keys allowed
for the resolved ``task_id`` are accepted when validating a run. Non-metric
telemetry keys (timing, diagnostics) are always permitted.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from vision_lab.metrics import STANDARD_METRICS
from vision_lab.task_registry import get_task

# Values kept as plain strings (not coerced to numbers).
STRING_SUMMARY_KEYS: frozenset[str] = frozenset({"task_type"})

# Resource and training diagnostics (never used as promotion primaries).
SUMMARY_TELEMETRY_KEYS: frozenset[str] = frozenset(
    {"training_seconds", "peak_vram_mb", "train_loss", "num_train_epochs"}
)

# Keys that should coerce to numbers when present (superset used when ``task_id`` is unknown).
NUMERIC_COERCION_KEYS: frozenset[str] = (
    frozenset(STANDARD_METRICS.keys()) | SUMMARY_TELEMETRY_KEYS | frozenset({"dice"})
)

_SUMMARY_LINE_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def accept_summary_line_key(key: str) -> bool:
    """True when ``key`` may appear left of ``:`` in a summary line."""
    return bool(_SUMMARY_LINE_KEY_RE.match(key))


def allowed_summary_keys_for_task(task_id: str) -> frozenset[str]:
    """All keys permitted in a summary block for ``task_id`` (including telemetry)."""
    spec = get_task(task_id)
    return (
        spec.promotion_metrics_union()
        | spec.allowed_auxiliary_summary_keys
        | SUMMARY_TELEMETRY_KEYS
        | STRING_SUMMARY_KEYS
    )


def validate_summary_keys_for_task(task_id: str, summary: Mapping[str, Any]) -> None:
    """Reject summary lines whose keys are not declared for the task contract."""
    allowed = allowed_summary_keys_for_task(task_id)
    bad = sorted(k for k in summary if k not in allowed)
    if bad:
        raise ValueError(
            f"Summary contains keys not allowed for task {task_id!r}: {bad}. "
            f"Allowed: {', '.join(sorted(allowed))}."
        )
