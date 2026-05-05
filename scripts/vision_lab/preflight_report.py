"""Shared preflight report for HF Jobs, local runs, and CI."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

from vision_lab.alignment_checks import collect_alignment_issues
from vision_lab.dataset_contracts import preflight_adapter_matches_task
from vision_lab.dataset_validation import validate_dataset
from vision_lab.task_registry import task_script_map

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_training_yaml(config_path: Path) -> dict[str, Any] | None:
    try:
        import yaml

        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def resolve_task_from_config(config_path: Path) -> str | None:
    cfg = load_training_yaml(config_path)
    return cfg.get("task_type") if cfg else None


def config_diff_preview(config_path: Path, task: str) -> tuple[list[str], int]:
    base_path = REPO_ROOT / "configs" / f"base_{task}.yaml"
    if not base_path.exists() or not config_path.exists():
        return [], 0
    base_lines = base_path.read_text(encoding="utf-8").splitlines()
    config_lines = config_path.read_text(encoding="utf-8").splitlines()
    diff_lines = list(
        difflib.unified_diff(
            base_lines,
            config_lines,
            fromfile=str(base_path.name),
            tofile=str(config_path.name),
            n=1,
            lineterm="",
        )
    )
    changed = sum(
        1
        for line in diff_lines
        if (line.startswith("+") or line.startswith("-"))
        and not line.startswith(("+++", "---"))
    )
    return diff_lines[:30], changed


def build_preflight_report(task: str, config_path: Path) -> dict[str, Any]:
    """Audit training config: runner script, YAML alignment (task/promotion/model), dataset adapter/path."""
    report: dict[str, Any] = {
        "task": task,
        "config": str(config_path),
        "errors": [],
        "warnings": [],
    }
    errors: list[str] = report["errors"]
    warnings: list[str] = report["warnings"]

    TASK_SCRIPTS = task_script_map()
    train_script = REPO_ROOT / TASK_SCRIPTS[task]
    if not train_script.exists():
        errors.append(f"missing {train_script.name}")
    if not config_path.exists():
        errors.append(f"missing config: {config_path}")
    else:
        diff_lines, changed = config_diff_preview(config_path, task)
        report["diff_preview"] = diff_lines
        report["diff_changed_lines"] = changed
        if changed == 0:
            warnings.append("config matches base; no experiment change")
        if changed > 20:
            warnings.append(
                "config differs by many lines; review for multi-change",
            )

        cfg = load_training_yaml(config_path)
        if cfg:
            ce, cw = collect_alignment_issues(cfg, task)
            errors.extend(ce)
            warnings.extend(cw)

            adapter_raw = cfg.get("dataset_adapter", "auto")
            adapter = adapter_raw if isinstance(adapter_raw, str) else "auto"
            ok_adp, adp_msg = preflight_adapter_matches_task(adapter, task)
            if not ok_adp and adp_msg:
                errors.append(f"dataset_adapter policy: {adp_msg}")

            dataset_root_raw = cfg.get("dataset_root")
            dataset_name_raw = cfg.get("dataset_name")

            validated_any = False
            if isinstance(dataset_root_raw, str) and dataset_root_raw.strip():
                dr = Path(dataset_root_raw).expanduser()
                if dr.exists():
                    vr = validate_dataset(
                        str(dr),
                        task,
                        split=str(cfg.get("dataset_split", "train")),
                        config=cfg.get("dataset_config_name"),
                        adapter_id=adapter,
                        write_cache=False,
                    )
                    validated_any = True
                    report["dataset_validation_path"] = str(dr)
                    report["dataset_validation_summary"] = {
                        "valid": vr.get("valid"),
                        "adapter_id": vr.get("adapter_id"),
                        "dataset_schema_kind": vr.get("dataset_schema_kind"),
                    }
                    if not vr.get("valid"):
                        for e in vr.get("errors") or []:
                            errors.append(f"dataset_root validation: {e}")
                    for w in vr.get("warnings") or []:
                        warnings.append(f"dataset_root: {w}")
                else:
                    warnings.append(
                        f"dataset_root {dataset_root_raw!r} not found on this machine "
                        "(skipped filesystem validation; may exist on the job runner)",
                    )

            if (
                not validated_any
                and isinstance(dataset_name_raw, str)
                and dataset_name_raw.strip()
            ):
                dn = Path(dataset_name_raw).expanduser()
                if dn.exists():
                    vr = validate_dataset(
                        str(dn),
                        task,
                        split=str(cfg.get("dataset_split", "train")),
                        config=cfg.get("dataset_config_name"),
                        adapter_id=adapter,
                        write_cache=False,
                    )
                    report["dataset_validation_path"] = str(dn)
                    report["dataset_validation_summary"] = {
                        "valid": vr.get("valid"),
                        "adapter_id": vr.get("adapter_id"),
                        "dataset_schema_kind": vr.get("dataset_schema_kind"),
                    }
                    if not vr.get("valid"):
                        for e in vr.get("errors") or []:
                            errors.append(f"dataset_name path validation: {e}")
                    for w in vr.get("warnings") or []:
                        warnings.append(f"dataset path: {w}")
                else:
                    warnings.append(
                        "dataset_name is not a local path; skipping Hub dataset download in preflight "
                        "(run prepare.py against the Hub id separately).",
                    )

    return report


def print_preflight_report(report: dict[str, Any]) -> None:
    print(f"Preflight: task={report['task']}")
    print(f"  config: {report['config']}")
    dv = report.get("dataset_validation_summary")
    if dv:
        print(f"  dataset_validation: {dv}")
    print(f"  changed_lines: {report.get('diff_changed_lines', 0)}")
    preview = report.get("diff_preview", [])
    if preview:
        print("  diff preview:")
        for line in preview:
            print(f"    {line}")
    for entry in report.get("errors", []):
        print(f"  ERROR: {entry}")
    for entry in report.get("warnings", []):
        print(f"  WARN: {entry}")
