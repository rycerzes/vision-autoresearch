"""Image-to-image pairs (e.g. restoration, synthesis)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vision_lab.dataset_contracts import AdapterPartialReport, finalize_local_report

_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_image_pairs(root: Path, *, max_check: int = 50) -> dict[str, Any]:
    root = root.resolve()
    errors: list[str] = []
    warnings: list[str] = []

    inp = next(
        (
            root / n
            for n in ("input", "lq", "src", "source", "noisy")
            if (root / n).is_dir()
        ),
        None,
    )
    out = next(
        (
            root / n
            for n in ("target", "hq", "gt", "reference", "clean")
            if (root / n).is_dir()
        ),
        None,
    )

    if inp is None or out is None:
        errors.append(
            f"Expected paired folders like input/ + target/ (or lq/ + hq/) under {root}"
        )

    matched = 0
    missing = 0
    if inp and out and inp.is_dir() and out.is_dir():
        inputs = sorted(
            p for p in inp.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXT
        )
        for im in inputs[:max_check]:
            if (out / im.name).is_file():
                matched += 1
                continue
            alt = next((p for p in out.glob(f"{im.stem}.*") if p.is_file()), None)
            if alt:
                matched += 1
            else:
                missing += 1

    if inp and out and matched == 0 and not errors:
        errors.append("No paired images found with matching names.")

    n_in = (
        len([p for p in inp.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXT])
        if inp and inp.is_dir()
        else 0
    )

    p = AdapterPartialReport(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        adapter_id="image_pairs",
        dataset_schema_kind="image_to_image",
        required_fields=["input/", "target/"],
        splits={"train": str(root)},
        row_counts={"input_images": n_in},
        inspection={
            "input_dir": str(inp) if inp else None,
            "target_dir": str(out) if out else None,
            "matched_sample": matched,
            "missing_sample": missing,
        },
    )
    return finalize_local_report(p)
