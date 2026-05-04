"""PASCAL VOC layout: ``JPEGImages`` + ``Annotations`` XML files."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from vision_lab.dataset_contracts import AdapterPartialReport, to_validation_report


def validate_voc_xml(root: Path, *, max_xml: int = 60) -> dict[str, Any]:
    root = root.resolve()
    jpeg = root / "JPEGImages"
    ann = root / "Annotations"
    errors: list[str] = []
    warnings: list[str] = []

    if not jpeg.is_dir():
        errors.append(f"Missing JPEGImages/ directory under {root}")
    if not ann.is_dir():
        errors.append(f"Missing Annotations/ directory under {root}")

    class_names: set[str] = set()
    paired = 0
    xml_files = sorted(ann.glob("*.xml"))[:max_xml] if ann.is_dir() else []

    for xf in xml_files:
        try:
            tree = ET.parse(xf)
            fn = tree.find("filename")
            if fn is not None and fn.text and jpeg.is_dir():
                img_path = jpeg / fn.text.strip()
                if not img_path.is_file():
                    warnings.append(f"{xf.name}: filename {fn.text!r} not in JPEGImages")
            for obj in tree.findall("object"):
                name_el = obj.find("name")
                if name_el is not None and name_el.text:
                    class_names.add(name_el.text.strip())
            stem = xf.stem
            if jpeg.is_dir():
                found = any((jpeg / f"{stem}{sfx}").is_file() for sfx in (".jpg", ".jpeg", ".png"))
                if found:
                    paired += 1
        except ET.ParseError as e:
            warnings.append(f"Bad XML {xf.name}: {e}")

    total_xml = len(list(ann.glob("*.xml"))) if ann.is_dir() else 0
    row_counts = {"train": total_xml}

    p = AdapterPartialReport(
        errors=errors,
        warnings=warnings,
        adapter_id="voc_xml",
        dataset_schema_kind="detection",
        required_fields=["JPEGImages/", "Annotations/*.xml"],
        detected_class_names=sorted(class_names),
        label_remapping={},
        splits={"train": str(root)},
        row_counts=row_counts,
        inspection={
            "xml_sample_paired": paired,
            "xml_total": total_xml,
            "warnings": warnings,
        },
    )
    return to_validation_report(p)
