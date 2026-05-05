"""Structured summary contract for the ``VISION AUTORESEARCH SUMMARY`` log block.

Trainers emit ``key: value`` lines between markers. Standard metrics (see
``vision_lab.metrics.STANDARD_METRICS``) get numeric coercion; other
identifier-shaped keys use best-effort numeric coercion (auxiliary logging only).
"""

from __future__ import annotations

import re

from vision_lab.metrics import STANDARD_METRICS

# Values kept as plain strings (not coerced to numbers).
STRING_SUMMARY_KEYS: frozenset[str] = frozenset({"task_type"})

REGISTERED_METRIC_KEYS: frozenset[str] = frozenset(STANDARD_METRICS.keys())

NUMERIC_SUMMARY_KEYS: frozenset[str] = REGISTERED_METRIC_KEYS

_SUMMARY_LINE_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def accept_summary_line_key(key: str) -> bool:
    """True when ``key`` may appear left of ``:`` in a summary line."""
    return bool(_SUMMARY_LINE_KEY_RE.match(key))
