#!/usr/bin/env python3
"""Parse the VISION AUTORESEARCH SUMMARY block from a training log."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import yaml

from vision_lab.promotion import assert_summary_eligible_for_recording, load_promotion_policy
from vision_lab.summary_schema import NUMERIC_COERCION_KEYS, STRING_SUMMARY_KEYS, accept_summary_line_key


def coerce_summary_value(raw: str, key: str) -> int | float | str:
    raw = raw.strip()
    if key in STRING_SUMMARY_KEYS:
        return raw
    if key in NUMERIC_COERCION_KEYS:
        for caster in (int, float):
            try:
                return caster(raw)
            except ValueError:
                continue
        return raw
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
            if not accept_summary_line_key(key):
                continue
            metrics[key] = coerce_summary_value(value, key)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse vision autoresearch metrics from a training log."
    )
    parser.add_argument("log_path", help="Path to the training log file")
    parser.add_argument("--key", help="Print only this metric value (no JSON wrapper)")
    parser.add_argument(
        "--task",
        help=(
            "If set with optional --config, validate summary metrics against the task promotion "
            "contract before printing."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="YAML config used to resolve promotion block when validating with --task",
    )
    args = parser.parse_args()

    text = Path(args.log_path).read_text(encoding="utf-8")
    metrics = parse_summary(text)

    if not metrics:
        print(
            f"No VISION AUTORESEARCH SUMMARY block found in {args.log_path}",
            file=sys.stderr,
        )
        return 1

    if args.task:
        cfg: dict = {}
        if args.config:
            loaded = yaml.safe_load(args.config.read_text(encoding="utf-8"))
            cfg = loaded if isinstance(loaded, dict) else {}
        try:
            policy = load_promotion_policy(cfg, task_id=args.task)
            assert_summary_eligible_for_recording(
                task_id=args.task,
                policy=policy,
                summary_metrics=metrics,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
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
