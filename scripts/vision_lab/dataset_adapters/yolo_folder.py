"""Ultralytics-style folder layout: images + labels (.txt) and optional ``classes.txt``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vision_lab.dataset_contracts import AdapterPartialReport, finalize_local_report

_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _collect_images(folder: Path) -> list[Path]:
    out: list[Path] = []
    if not folder.is_dir():
        return out
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in _IMAGE_EXT:
            out.append(p)
    return out


def validate_yolo_folder(root: Path, *, max_pairs_check: int = 40) -> dict[str, Any]:
    root = root.resolve()
    errors: list[str] = []
    warnings: list[str] = []

    images_dir: Path | None = None
    labels_dir: Path | None = None
    for cand in (
        root / "images" / "train",
        root / "train" / "images",
        root / "images",
        root / "JPEGImages",
        root,
    ):
        if cand.is_dir():
            imgs = _collect_images(cand)
            if len(imgs) >= 1:
                images_dir = cand
                break

    if images_dir is None:
        p = AdapterPartialReport(
            valid=False,
            errors=[f"No image files found under {root} (expected images/train or images/)."],
            adapter_id="yolo_folder",
            dataset_schema_kind="detection",
            required_fields=["images/", "labels/*.txt"],
        )
        return finalize_local_report(p)

    labels_dir = None
    if (root / "labels").is_dir():
        if images_dir.is_relative_to(root):
            rel = images_dir.relative_to(root)
            if rel.parts:
                cand = root / "labels" / rel.parts[0]
                if cand.is_dir():
                    labels_dir = cand
    if labels_dir is None:
        for cand in (root / "labels" / "train", root / "train" / "labels", root / "labels"):
            if cand.is_dir():
                labels_dir = cand
                break

    if labels_dir is None:
        errors.append(
            "Could not find parallel labels/ directory with .txt files for YOLO detection."
        )

    class_names: list[str] = []
    classes_file = root / "classes.txt"
    if classes_file.is_file():
        class_names = [
            line.strip()
            for line in classes_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif (root / "data.yaml").is_file():
        warnings.append("data.yaml present — class names may be defined there (not parsed here).")

    missing_labels = 0
    nonempty_labels = 0
    images = _collect_images(images_dir)[:max_pairs_check]
    for img in images:
        stem = img.stem
        if labels_dir:
            lf = labels_dir / f"{stem}.txt"
            if not lf.is_file():
                missing_labels += 1
            else:
                txt = lf.read_text(encoding="utf-8").strip()
                if txt:
                    nonempty_labels += 1

    if labels_dir and missing_labels == len(images) and images:
        errors.append(f"No matching label .txt files in {labels_dir} for sampled images.")

    row_counts = {"train": len(_collect_images(images_dir))}

    label_remapping = {str(i): n for i, n in enumerate(class_names)}
    inspection = {
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir) if labels_dir else None,
        "classes_txt": bool(classes_file.is_file()),
        "sample_checked": len(images),
        "missing_labels_sample": missing_labels,
        "nonempty_labels_sample": nonempty_labels,
        "warnings": warnings,
    }

    p = AdapterPartialReport(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        adapter_id="yolo_folder",
        dataset_schema_kind="detection",
        required_fields=["images/", "labels/*.txt"],
        detected_class_names=class_names,
        label_remapping=label_remapping,
        splits={"train": str(images_dir)},
        row_counts=row_counts,
        inspection=inspection,
    )
    return finalize_local_report(p)
