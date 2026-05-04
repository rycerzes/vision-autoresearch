"""Write reproducible cache manifests under ``.runtime/datasets/`` for validated layouts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_ROOT = REPO_ROOT / ".runtime" / "datasets"


def fingerprint_local_tree(root: Path, *, max_paths: int = 800) -> str:
    """
    Stable fingerprint from relative paths and file sizes (bounded work).

    Sorted walks avoid filesystem order ambiguity.
    """
    root = root.resolve()
    lines: list[str] = []
    count = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(root).as_posix()
            st = path.stat()
            lines.append(f"{rel}\t{st.st_size}")
        except OSError:
            continue
        count += 1
        if count >= max_paths:
            lines.append("__truncated__\t1")
            break
    payload = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_cache_manifest(
    *,
    source_path: Path,
    adapter_id: str,
    fingerprint: str,
    report_subset: dict[str, Any],
    cache_root: Path | None = None,
) -> Path:
    """Write ``manifest.json`` under a fingerprinted directory; return manifest path."""
    base = cache_root or DEFAULT_CACHE_ROOT
    short = fingerprint[:24]
    safe_adapter = "".join(c if c.isalnum() else "_" for c in adapter_id)
    dest = base / f"{safe_adapter}_{short}"
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "adapter_id": adapter_id,
        "fingerprint_sha256": fingerprint,
        "source_path": str(source_path.resolve()),
        "report": report_subset,
    }
    out_path = dest / "manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path
