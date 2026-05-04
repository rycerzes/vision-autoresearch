"""Orchestrate dataset validation across HF Hub and filesystem adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from vision_lab.dataset_adapters.coco_json import find_coco_json, validate_coco_json
from vision_lab.dataset_adapters.depth_pairs import validate_depth_pairs
from vision_lab.dataset_adapters.hf_hub import run_hf_hub_adapter
from vision_lab.dataset_adapters.image_pairs import validate_image_pairs
from vision_lab.dataset_adapters.ocr_table import validate_ocr_table
from vision_lab.dataset_adapters.sam_prompt_mask import validate_sam_prompt_mask
from vision_lab.dataset_adapters.semantic_masks import validate_semantic_masks
from vision_lab.dataset_adapters.video_folder import validate_video_folder
from vision_lab.dataset_adapters.voc_xml import validate_voc_xml
from vision_lab.dataset_adapters.yolo_folder import validate_yolo_folder
from vision_lab.dataset_cache import fingerprint_local_tree, write_cache_manifest
from vision_lab.dataset_contracts import ADAPTER_SCHEMA_KIND, EXTENDED_SCHEMA_KINDS
from vision_lab.task_registry import TASK_BY_ID

NUM_INSPECT_SAMPLES_DEFAULT = 5

_LOCAL_VALIDATORS: dict[str, Callable[..., dict[str, Any]]] = {
    "coco_json": validate_coco_json,
    "yolo_folder": validate_yolo_folder,
    "voc_xml": validate_voc_xml,
    "semantic_masks": validate_semantic_masks,
    "sam_prompt_mask": validate_sam_prompt_mask,
    "video_folder": validate_video_folder,
    "ocr_table": validate_ocr_table,
    "depth_pairs": validate_depth_pairs,
    "image_pairs": validate_image_pairs,
}


def _looks_like_coco_json_file(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() != ".json":
        return False
    try:
        head = path.read_text(encoding="utf-8")[:4096]
        return '"images"' in head and '"annotations"' in head and '"categories"' in head
    except OSError:
        return False


def infer_local_adapter(source: Path) -> str | None:
    """Pick a filesystem adapter using cheap structural probes (first match wins)."""
    root = source.resolve()
    if root.is_file():
        return "coco_json" if _looks_like_coco_json_file(root) else None

    if (root / "JPEGImages").is_dir() and (root / "Annotations").is_dir():
        return "voc_xml"

    if any((root / n).is_dir() for n in ("rgb", "images", "color")) and (root / "depth").is_dir():
        return "depth_pairs"

    if any((root / n).is_dir() for n in ("input", "lq", "src")) and any(
        (root / n).is_dir() for n in ("target", "hq", "gt")
    ):
        return "image_pairs"

    if find_coco_json(root):
        return "coco_json"

    imgs_train = root / "images" / "train"
    train_img = root / "train" / "images"
    if (
        imgs_train.is_dir()
        or train_img.is_dir()
        or (root / "images").is_dir()
        or (root / "labels").is_dir()
    ):
        probe = validate_yolo_folder(root)
        if probe.get("valid"):
            return "yolo_folder"

    img_dir = root / "images" if (root / "images").is_dir() else root / "img"
    mask_side = root / "masks" if (root / "masks").is_dir() else root / "labels"
    if img_dir.is_dir() and mask_side.is_dir():
        if any(p.is_file() for p in (root / "prompts.jsonl", root / "annotations.jsonl")):
            return "sam_prompt_mask"
        sm = validate_semantic_masks(root)
        if sm.get("valid"):
            return "semantic_masks"

    vf = validate_video_folder(root)
    if vf.get("valid"):
        return "video_folder"

    if any((root / n).is_file() for n in ("gt.json", "manifest.csv", "train.jsonl", "ocr_gt.jsonl")):
        ot = validate_ocr_table(root)
        if ot.get("valid"):
            return "ocr_table"

    return None


def _enforce_registered_task_schema(report: dict[str, Any], task_type: str) -> None:
    """Ensure ``report`` layout matches ``task_type`` for registered (non-extended) schemas."""
    if task_type not in TASK_BY_ID:
        return
    kind = report.get("dataset_schema_kind") or ""
    if kind in EXTENDED_SCHEMA_KINDS:
        report.setdefault("errors", []).append(
            f"Dataset layout uses extended schema {kind!r} which has no matching registered "
            "task in task_registry yet."
        )
        report["valid"] = False
        return
    compat = report.get("compatible_tasks") or []
    if compat and task_type not in compat:
        report.setdefault("errors", []).append(
            f"Task {task_type!r} does not match adapter {report.get('adapter_id')!r} layout "
            f"(compatible tasks: {compat})."
        )
        report["valid"] = False


def validate_dataset(
    dataset_name: str,
    task_type: str,
    split: str = "train",
    config: str | None = None,
    inspect: bool = False,
    num_samples: int = NUM_INSPECT_SAMPLES_DEFAULT,
    *,
    adapter_id: str = "auto",
    write_cache: bool = True,
) -> dict[str, Any]:
    """
    Validate dataset for ``task_type``.

    When ``dataset_name`` is an existing file or directory and ``adapter_id`` is ``auto``,
    filesystem adapters are inferred; otherwise ``hf_hub`` loads from the Hub.

    Extended report keys: ``adapter_id``, ``dataset_schema_kind``, ``compatible_tasks``, ``warnings``,
    ``cache_manifest_path`` (optional).
    """
    src = Path(dataset_name).expanduser()
    local_ok = src.exists()

    aid = adapter_id.strip() if adapter_id else "auto"
    use_hf = aid == "hf_hub" or (aid == "auto" and not local_ok)

    if aid not in ("auto", "hf_hub") and not local_ok:
        return {
            "valid": False,
            "errors": [
                f"Adapter {aid!r} expects an existing filesystem path; {dataset_name!r} not found.",
            ],
            "warnings": [],
            "adapter_id": aid,
            "dataset_schema_kind": "",
            "compatible_tasks": [],
            "required_fields": [],
            "detected_class_names": [],
            "label_remapping": {},
            "splits": {},
            "row_counts": {},
            "columns": [],
            "num_rows": 0,
            "config": config,
            "inspection": None,
            "cache_manifest_path": None,
        }

    if use_hf:
        report = run_hf_hub_adapter(
            dataset_name,
            task_type,
            split=split,
            config=config,
            inspect=inspect,
            num_samples=num_samples,
            task_by_id=TASK_BY_ID,
        )
        report.setdefault("cache_manifest_path", None)
        _enforce_registered_task_schema(report, task_type)
        return report

    resolved = infer_local_adapter(src) if aid == "auto" else aid
    if resolved is None:
        return {
            "valid": False,
            "errors": [
                f"Could not infer dataset adapter for path {src}. "
                f"Pass --adapter with one of: {', '.join(sorted(_LOCAL_VALIDATORS))}.",
            ],
            "warnings": [],
            "adapter_id": "auto",
            "dataset_schema_kind": "",
            "compatible_tasks": [],
            "required_fields": [],
            "detected_class_names": [],
            "label_remapping": {},
            "splits": {},
            "row_counts": {},
            "columns": [],
            "num_rows": -1,
            "config": config,
            "inspection": {"hint": "Try explicit --adapter or organize files per dataset_adapters modules."},
            "cache_manifest_path": None,
        }

    if resolved not in _LOCAL_VALIDATORS:
        return {
            "valid": False,
            "errors": [f"Unknown local adapter: {resolved!r}"],
            "warnings": [],
            "adapter_id": resolved,
            "dataset_schema_kind": ADAPTER_SCHEMA_KIND.get(resolved, ""),
            "compatible_tasks": [],
            "required_fields": [],
            "detected_class_names": [],
            "label_remapping": {},
            "splits": {},
            "row_counts": {},
            "columns": [],
            "num_rows": -1,
            "config": config,
            "inspection": None,
            "cache_manifest_path": None,
        }

    runner = _LOCAL_VALIDATORS[resolved]
    if resolved == "coco_json" and src.is_file():
        report = validate_coco_json(src)
    else:
        report = runner(src)

    report["adapter_id"] = resolved

    if write_cache and report.get("valid"):
        try:
            fp_base = src if src.is_dir() else src.parent
            fp = fingerprint_local_tree(fp_base)
            manifest_path = write_cache_manifest(
                source_path=fp_base,
                adapter_id=resolved,
                fingerprint=fp,
                report_subset={
                    "dataset_schema_kind": report.get("dataset_schema_kind"),
                    "row_counts": report.get("row_counts"),
                    "compatible_tasks": report.get("compatible_tasks"),
                },
            )
            report["cache_manifest_path"] = str(manifest_path)
        except OSError:
            report.setdefault("warnings", []).append("Could not write dataset cache manifest.")
            report.setdefault("cache_manifest_path", None)

    _enforce_registered_task_schema(report, task_type)
    return report


def local_adapter_ids() -> tuple[str, ...]:
    return tuple(sorted(_LOCAL_VALIDATORS.keys()))


def all_adapter_ids_cli() -> tuple[str, ...]:
    return tuple(sorted({"auto", "hf_hub", *local_adapter_ids()}))
