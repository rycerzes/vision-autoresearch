#!/usr/bin/env python3
"""Parse the VISION AUTORESEARCH SUMMARY block from a training log."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


SUMMARY_KEYS = {
    "task_type",
    "mAP",
    "mAP_50",
    "mAR",
    "accuracy",
    "iou",
    "dice",
    "training_seconds",
    "peak_vram_mb",
    "train_loss",
    "num_train_epochs",
}

NUMERIC_KEYS = {
    "mAP",
    "mAP_50",
    "mAR",
    "accuracy",
    "iou",
    "dice",
    "training_seconds",
    "peak_vram_mb",
    "train_loss",
    "num_train_epochs",
}


def coerce_value(raw: str, key: str) -> int | float | str:
    raw = raw.strip()
    if key in NUMERIC_KEYS:
        for caster in (int, float):
            try:
                return caster(raw)
            except ValueError:
                continue
    return raw


def parse_summary(text: str) -> dict[str, int | float | str]:
    metrics: dict[str, int | float | str] = {}
    in_summary = False
    for line in text.splitlines():
        stripped = line.strip()
        if "VISION AUTORESEARCH SUMMARY" in stripped:
            in_summary = True
            continue
        if "END SUMMARY" in stripped:
            break
        if not in_summary:
            continue
        match = re.match(r"^([A-Za-z_0-9]+):\s+(.+)$", stripped)
        if match:
            key, value = match.groups()
            if key in SUMMARY_KEYS:
                metrics[key] = coerce_value(value, key)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse vision autoresearch metrics from a training log."
    )
    parser.add_argument("log_path", help="Path to the training log file")
    parser.add_argument("--key", help="Print only this metric value (no JSON wrapper)")
    args = parser.parse_args()

    text = Path(args.log_path).read_text(encoding="utf-8")
    metrics = parse_summary(text)

    if not metrics:
        print(
            f"No VISION AUTORESEARCH SUMMARY block found in {args.log_path}",
            file=sys.stderr,
        )
        return 1

    if args.key:
        value = metrics.get(args.key)
        if value is None:
            print(f"Key '{args.key}' not found in summary", file=sys.stderr)
            return 1
        print(value)
        return 0

    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
