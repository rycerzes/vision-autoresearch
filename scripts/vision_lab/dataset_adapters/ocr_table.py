"""OCR / table recognition layouts: images plus ground-truth text files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from vision_lab.dataset_contracts import AdapterPartialReport, finalize_local_report

_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def validate_ocr_table(root: Path, *, max_rows: int = 80) -> dict[str, Any]:
    root = root.resolve()
    errors: list[str] = []
    warnings: list[str] = []

    gt_candidates = [
        root / "gt.json",
        root / "labels.json",
        root / "ocr_gt.jsonl",
        root / "train.jsonl",
        root / "manifest.csv",
    ]
    gt_path = next((p for p in gt_candidates if p.is_file()), None)

    texts_found = 0
    rows = 0
    if gt_path:
        if gt_path.suffix.lower() == ".csv":
            try:
                with gt_path.open(encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for i, row in enumerate(reader):
                        if i >= max_rows:
                            break
                        rows += 1
                        if any(k.lower() in ("text", "label", "transcription") for k in row):
                            texts_found += 1
            except OSError as e:
                errors.append(f"Could not read CSV {gt_path}: {e}")
        elif gt_path.suffix.lower() == ".jsonl":
            try:
                lines = gt_path.read_text(encoding="utf-8").splitlines()[:max_rows]
                for line in lines:
                    rows += 1
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict) and (
                            "text" in obj or "label" in obj or "html" in obj or "cells" in obj
                        ):
                            texts_found += 1
                    except json.JSONDecodeError:
                        pass
            except OSError as e:
                errors.append(f"Could not read JSONL {gt_path}: {e}")
        elif gt_path.suffix.lower() == ".json":
            try:
                data = json.loads(gt_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    rows = min(len(data), max_rows)
                    texts_found = sum(
                        1
                        for item in data[:max_rows]
                        if isinstance(item, dict)
                        and ("text" in item or "label" in item or "cells" in item)
                    )
                elif isinstance(data, dict):
                    rows = 1
                    texts_found = int(
                        "annotations" in data or "images" in data or "html" in data
                    )
            except (OSError, json.JSONDecodeError) as e:
                errors.append(f"Could not read JSON {gt_path}: {e}")
    else:
        warnings.append("No gt.json / manifest.csv / train.jsonl found — expected OCR/table GT.")

    imgs = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXT)
    if not imgs:
        errors.append(f"No images found under {root}")

    if gt_path and texts_found == 0 and rows > 0:
        warnings.append(
            f"{gt_path.name}: read {rows} rows but no obvious text/cells fields detected."
        )

    row_counts = {"images": len(imgs), "gt_rows_sampled": rows}

    p = AdapterPartialReport(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        adapter_id="ocr_table",
        dataset_schema_kind="ocr",
        required_fields=["images", "gt.json or manifest.csv"],
        splits={"train": str(root)},
        row_counts=row_counts,
        inspection={
            "gt_file": str(gt_path) if gt_path else None,
            "text_like_rows": texts_found,
            "warnings": warnings,
        },
    )
    return finalize_local_report(p)
