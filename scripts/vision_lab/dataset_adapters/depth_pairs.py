"""RGB / depth image pairs for depth estimation tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vision_lab.dataset_contracts import AdapterPartialReport, finalize_local_report

_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_depth_pairs(root: Path, *, max_check: int = 60) -> dict[str, Any]:
    root = root.resolve()
    errors: list[str] = []
    warnings: list[str] = []

    rgb_dir = next(
        (root / n for n in ("rgb", "images", "color", "image") if (root / n).is_dir()), None
    )
    depth_dir = next(
        (root / n for n in ("depth", "depths", "disparity", "target") if (root / n).is_dir()),
        None
    )

    if rgb_dir is None:
        errors.append(f"No rgb/ or images/ directory under {root}")
    if depth_dir is None:
        errors.append(f"No depth/ directory under {root}")

    matched = 0
    missing = 0
    rgb_files: list[Path] = []
    if rgb_dir and rgb_dir.is_dir():
        rgb_files = sorted(
            p for p in rgb_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXT
        )

    if rgb_dir and depth_dir:
        for im in rgb_files[:max_check]:
            stem = im.stem
            found = False
            for ext in (".png", ".jpg", ".jpeg", ".tif", ".exr", ".npy"):
                if (depth_dir / f"{stem}{ext}").is_file():
                    found = True
                    break
            if found:
                matched += 1
            else:
                missing += 1

    if rgb_files and matched == 0 and not errors:
        errors.append("No depth file matched RGB stems in depth/.")

    row_counts = {"pairs_estimated": len(rgb_files)}
    p = AdapterPartialReport(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        adapter_id="depth_pairs",
        dataset_schema_kind="depth",
        required_fields=["rgb/", "depth/"],
        splits={"train": str(root)},
        row_counts=row_counts,
        inspection={
            "rgb_dir": str(rgb_dir) if rgb_dir else None,
            "depth_dir": str(depth_dir) if depth_dir else None,
            "matched_sample": matched,
            "missing_sample": missing,
        },
    )
    return finalize_local_report(p)
