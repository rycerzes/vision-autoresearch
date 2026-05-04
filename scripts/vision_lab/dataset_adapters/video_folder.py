"""Video clips or frame folders for future video tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vision_lab.dataset_contracts import AdapterPartialReport, to_validation_report

_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_IMAGE_EXT = {".jpg", ".jpeg", ".png"}


def validate_video_folder(root: Path, *, max_list: int = 200) -> dict[str, Any]:
    root = root.resolve()
    errors: list[str] = []
    warnings: list[str] = []

    videos = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _VIDEO_EXT
    )[:max_list]

    frame_dirs = [p for p in root.iterdir() if p.is_dir()][:50]
    nested_frames = 0
    for d in frame_dirs:
        imgs = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXT]
        if len(imgs) >= 3:
            nested_frames += 1

    if not videos and nested_frames == 0:
        errors.append(
            f"No video files ({sorted(_VIDEO_EXT)}) or frame subfolders under {root}"
        )

    row_counts = {"videos": len(videos), "frame_like_dirs": nested_frames}
    kind = "video"

    p = AdapterPartialReport(
        errors=errors,
        warnings=warnings,
        adapter_id="video_folder",
        dataset_schema_kind=kind,
        required_fields=["*.mp4 or frames/*/"],
        splits={"root": str(root)},
        row_counts=row_counts,
        inspection={
            "video_sample_count": len(videos),
            "frame_directories_hint": nested_frames,
            "warnings": warnings,
        },
    )
    return to_validation_report(p)
