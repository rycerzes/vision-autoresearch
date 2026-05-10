"""HF Hub dataset inspection: hub metadata, declared splits, features, row previews."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset
from huggingface_hub import dataset_info as hf_dataset_info

from vision_lab.contracts.run_contract import contract_fingerprint_from_payload

JSONDict = dict[str, Any]


def profile_to_primitive_dict(profile: HubDatasetProfile) -> dict[str, Any]:
    """Serialize ``HubDatasetProfile`` to nested dicts/lists for JSON or hashing."""
    return _dataclass_to_primitive(profile)


def profile_fingerprint(profile: HubDatasetProfile) -> str:
    """Stable digest of the inspection artifact (canonical JSON)."""
    return contract_fingerprint_from_payload(profile_to_primitive_dict(profile))  # type: ignore[arg-type]


def _dataclass_to_primitive(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_primitive(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _dataclass_to_primitive(v) for k, v in sorted(obj.items())}
    if is_dataclass(obj) and not isinstance(obj, type):
        out: dict[str, Any] = {}
        for f in fields(obj):
            out[f.name] = _dataclass_to_primitive(getattr(obj, f.name))
        return out
    raise TypeError(f"Unsupported type for profile serialization: {type(obj)!r}")


def _feature_summary(obj: Any) -> JSONDict:
    """JSON-serializable summary of a ``datasets`` Feature."""
    from datasets import ClassLabel, Image, Sequence, Value

    if isinstance(obj, Value):
        return {"kind": "Value", "dtype": str(obj.dtype)}
    if isinstance(obj, ClassLabel):
        names = getattr(obj, "names", None) or []
        return {"kind": "ClassLabel", "num_classes": len(names)}
    if isinstance(obj, Image):
        return {
            "kind": "Image",
            "decode": bool(getattr(obj, "decode", True)),
        }
    if isinstance(obj, Sequence):
        inner = getattr(obj, "feature", None)
        return {"kind": "Sequence", "feature": _feature_summary(inner) if inner is not None else {}}
    cls_name = obj.__class__.__name__
    if hasattr(obj, "dtype") and hasattr(obj, "__class__"):
        try:
            return {"kind": cls_name, "dtype": str(getattr(obj, "dtype", ""))}
        except Exception:
            return {"kind": cls_name}
    return {"kind": cls_name}


def _features_schema(features: Any) -> tuple[tuple[str, JSONDict], ...]:
    keys = sorted(features.keys()) if hasattr(features, "keys") else []
    return tuple((name, _feature_summary(features[name])) for name in keys)


def _resolve_config_name(
    repo_id: str, revision: str | None, requested: str | None, token: str | None
) -> tuple[str | None, list[str]]:
    errors: list[str] = []
    try:
        names = get_dataset_config_names(repo_id, revision=revision, token=token)
    except Exception as e:
        return None, [f"config enumeration failed: {e}"]
    if not names:
        return None, ["dataset reports no configs"]
    if requested is not None:
        if requested not in names:
            errors.append(f"config_name {requested!r} not in {sorted(names)}")
            return None, errors
        return requested, []
    if len(names) == 1:
        return names[0], []
    if "default" in names:
        return "default", []
    errors.append(
        "multiple dataset configs available; pass explicit config_name "
        f"(candidates: {', '.join(sorted(names))})"
    )
    return None, errors


def _load_hub_slice(
    repo_id: str,
    config_name: str | None,
    split: str,
    revision: str | None,
    num_samples: int,
    token: str | None,
) -> tuple[Any, list[dict[str, Any]], list[str]]:
    """Return (features_dict_like, samples, errors)."""
    errors: list[str] = []
    slice_split = f"{split}[:{num_samples}]"
    common_kw: dict[str, Any] = {"revision": revision, "token": token}
    try:
        ds = load_dataset(repo_id, config_name, split=slice_split, streaming=False, **common_kw)
        n = min(num_samples, len(ds))
        samples = [ds[i] for i in range(n)]
        feats = ds.features
        return feats, samples, errors
    except Exception:
        pass
    try:
        ds = load_dataset(repo_id, config_name, split=split, streaming=True, **common_kw)
        samples = []
        for i, row in enumerate(ds):
            samples.append(dict(row))
            if i + 1 >= num_samples:
                break
        feats = ds.features
        return feats, samples, errors
    except Exception as e:
        errors.append(f"dataset load failed: {e}")
        return None, [], errors


def _row_keys(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(row.keys()))


@dataclass(frozen=True)
class HubDatasetProfile:
    """Normalized HF Hub inspection artifact (reproducible inputs for resolvers)."""

    repo_id: str
    revision_requested: str | None
    revision_resolved: str
    config_name: str | None
    split: str
    splits_available: tuple[str, ...]
    feature_schema: tuple[tuple[str, JSONDict], ...]
    sample_row_keys: tuple[tuple[str, ...], ...]
    num_samples_inspected: int
    hard_errors: tuple[str, ...]
    soft_warnings: tuple[str, ...]
    inspection_confidence: float
    inspection_rationale: str


def inspect_hf_hub_dataset(
    repo_id: str,
    *,
    revision: str | None,
    config_name: str | None,
    split: str,
    num_samples: int = 5,
    token: str | None = None,
) -> HubDatasetProfile:
    """
    Load hub metadata and a small split slice; emit a strict, fingerprintable profile.

    ``revision`` may be ``None`` to follow the Hub default branch; ``revision_resolved``
    in the profile is always the commit sha returned by the Hub API for auditing.
    """
    hard: list[str] = []
    soft: list[str] = []

    try:
        info = hf_dataset_info(repo_id=repo_id, revision=revision, token=token)
        rev_out = str(getattr(info, "sha", "") or "").strip()
        if not rev_out:
            rev_out = (revision or "").strip()
            if not rev_out:
                hard.append("hub metadata did not return a resolved revision sha")
    except Exception as e:
        hard.append(f"hub metadata failed: {e}")
        rev_out = (revision or "").strip()

    resolved_config, cfg_errors = _resolve_config_name(repo_id, revision, config_name, token)
    hard.extend(cfg_errors)

    splits_avail: tuple[str, ...] = ()
    if not hard and resolved_config is not None:
        try:
            raw_splits = get_dataset_split_names(
                repo_id,
                config_name=resolved_config,
                revision=revision,
                token=token,
            )
            splits_avail = tuple(sorted(set(raw_splits)))
        except Exception as e:
            hard.append(f"split enumeration failed: {e}")

    if splits_avail and split not in splits_avail:
        hard.append(f"split {split!r} not in declared splits {list(splits_avail)}")

    feats: Any = None
    samples: list[dict[str, Any]] = []
    if not hard and resolved_config is not None:
        feats, samples, load_err = _load_hub_slice(
            repo_id, resolved_config, split, revision, num_samples, token
        )
        hard.extend(load_err)
        if feats is None and not load_err:
            hard.append("features unavailable after load")

    feature_schema: tuple[tuple[str, JSONDict], ...] = ()
    sample_keys: tuple[tuple[str, ...], ...] = ()
    n_insp = 0
    if feats is not None and hasattr(feats, "keys"):
        try:
            feature_schema = _features_schema(feats)
        except Exception as e:
            hard.append(f"feature schema serialization failed: {e}")
        sample_keys = tuple(_row_keys(s) for s in samples)
        n_insp = len(samples)

    if revision is None and rev_out and not hard:
        soft.append(
            "revision was unset; profile.revision_resolved pins the current Hub commit for reproducibility"
        )

    confidence = 0.0 if hard else (1.0 if n_insp > 0 else 0.4)
    rationale_parts = [
        f"repo={repo_id!r} split={split!r} config={resolved_config!r}",
        f"samples_inspected={n_insp}",
    ]
    if hard:
        rationale_parts.append("status=incomplete")
    else:
        rationale_parts.append("status=ok")
    rationale = "; ".join(rationale_parts)

    return HubDatasetProfile(
        repo_id=repo_id,
        revision_requested=revision,
        revision_resolved=rev_out,
        config_name=resolved_config,
        split=split,
        splits_available=splits_avail,
        feature_schema=feature_schema,
        sample_row_keys=sample_keys,
        num_samples_inspected=n_insp,
        hard_errors=tuple(hard),
        soft_warnings=tuple(soft),
        inspection_confidence=confidence,
        inspection_rationale=rationale,
    )
