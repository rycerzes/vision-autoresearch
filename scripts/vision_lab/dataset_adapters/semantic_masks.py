"""Parallel ``images/`` and ``masks/`` (or ``labels/``) folders with matching stems."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vision_lab.dataset_contracts import AdapterPartialReport, to_validation_report

_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_MASK_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _list_images(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted(
        p for p in d.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXT
    )


def _find_mask(mask_dir: Path, stem: str) -> Path | None:
    for ext in _MASK_EXT:
        p = mask_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def validate_semantic_masks(root: Path, *, max_check: int = 50) -> dict[str, Any]:
    root = root.resolve()
    errors: list[str] = []
    warnings: list[str] = []

    img_dir = root / "images" if (root / "images").is_dir() else root / "img"
    m_dir = (
        root / "masks"
        if (root / "masks").is_dir()
        else root / "labels"
        if (root / "labels").is_dir()
        else root / "mask"
    )

    if not img_dir.is_dir():
        errors.append(f"Missing images/ (or img/) under {root}")
    if not m_dir.is_dir():
        errors.append(f"Missing masks/ (or labels/) under {root}")

    matches = 0
    missing = 0
    images = _list_images(img_dir) if img_dir.is_dir() else []
    for im in images[:max_check]:
        m = _find_mask(m_dir, im.stem) if m_dir.is_dir() else None
        if m:
            matches += 1
        else:
            missing += 1

    if images and matches == 0 and not errors:
        errors.append("No mask file found with matching image stem.")

    row_counts = {"train": len(images)}
    inspection = {
        "images_dir": str(img_dir),
        "masks_dir": str(m_dir),
        "matched_sample": matches,
        "missing_sample": missing,
    }
    p = AdapterPartialReport(
        errors=errors,
        warnings=warnings,
        adapter_id="semantic_masks",
        dataset_schema_kind="semantic_segmentation",
        required_fields=["images/", "masks/"],
        detected_class_names=[],
        splits={"train": str(img_dir)},
        row_counts=row_counts,
        inspection=inspection,
    )
    return to_validation_report(p)
