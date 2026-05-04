"""Structured summary contract for the ``VISION AUTORESEARCH SUMMARY`` log block.

Trainers emit ``key: value`` lines between markers. Registered metrics (see
``vision_lab.metrics.METRICS``) define numeric coercion for known keys; any other
identifier-shaped key is parsed with best-effort numeric coercion.
"""

from __future__ import annotations

import re

from vision_lab.metrics import METRICS

# Values kept as plain strings (not coerced to numbers).
STRING_SUMMARY_KEYS: frozenset[str] = frozenset({"task_type"})

REGISTERED_METRIC_KEYS: frozenset[str] = frozenset(METRICS.keys())

NUMERIC_SUMMARY_KEYS: frozenset[str] = REGISTERED_METRIC_KEYS

_SUMMARY_LINE_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def accept_summary_line_key(key: str) -> bool:
    """True when ``key`` may appear left of ``:`` in a summary line."""
    return bool(_SUMMARY_LINE_KEY_RE.match(key))
