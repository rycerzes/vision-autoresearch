"""Unified dataset abstraction — wraps HF datasets and exports to any format.

Works with both HF Trainer and Ultralytics by converting to each backend's
expected format on demand.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _find_class_label_names(features: Any, depth: int = 0) -> list[str] | None:
    """Recursively scan HF features for the first ClassLabel with names."""
    if depth > 5:
        return None

    # Direct ClassLabel
    if hasattr(features, "names") and features.names:
        return list(features.names)

    # Unwrap .feature (Sequence / List)
    inner = getattr(features, "feature", None)
    if inner is not None:
        result = _find_class_label_names(inner, depth + 1)
        if result:
            return result

    # Dict-like: iterate all values
    items = None
    if isinstance(features, dict):
        items = features.values()
    elif hasattr(features, "keys") and callable(features.keys):
        try:
            keys = getattr(features, "keys")()
            items = [features[k] for k in list(keys)]
        except Exception:
            pass
    elif hasattr(features, "items"):
        try:
            items = [v for _, v in features.items()]
        except Exception:
            pass

    if items is not None:
        for sub in items:
            result = _find_class_label_names(sub, depth + 1)
            if result:
                return result

    return None


class UnifiedDataset:
    """Wraps an HF dataset and can export to any backend format.

    The dataset is loaded lazily and column mapping is done once via
    :meth:`auto_map_columns` or by setting ``column_map`` directly.
    """

    def __init__(
        self,
        dataset_name: str,
        config: str | None = None,
        *,
        trust_remote_code: bool = True,
        cache_dir: str | None = None,
    ) -> None:
        from datasets import load_dataset

        self._dataset_name = dataset_name
        self._config = config
        try:
            self.hf_dataset = load_dataset(
                dataset_name,
                config,
                trust_remote_code=trust_remote_code,
                cache_dir=cache_dir,
            )
        except TypeError:
            # Newer datasets versions may reject trust_remote_code
            self.hf_dataset = load_dataset(
                dataset_name,
                config,
                cache_dir=cache_dir,
            )
        self.column_map: dict[str, str] | None = None
        self._class_names: list[str] | None = None
        logger.info(
            "UnifiedDataset loaded: %s  splits=%s",
            dataset_name,
            list(self.hf_dataset.keys()),
        )


    def auto_map_columns(
        self, head_category: str, processor: Any = None
    ) -> dict[str, str]:
        """Type-based column alignment. Stores result in ``self.column_map``."""
        from engine.column_mapper import auto_map_columns

        train_split = self._get_train_split()
        features = self.hf_dataset[train_split].features
        self.column_map = auto_map_columns(
            dict(features), head_category, processor=processor
        )
        return self.column_map

    def set_column_map(self, column_map: dict[str, str]) -> None:
        """Explicit override from YAML config."""
        self.column_map = dict(column_map)


    @property
    def class_names(self) -> list[str] | None:
        """Discover class names from dataset features.  No hardcoded column names —
        scans ALL columns for ClassLabel features at any nesting depth."""
        if self._class_names is not None:
            return self._class_names

        train_split = self._get_train_split()
        features = self.hf_dataset[train_split].features

        found = _find_class_label_names(features)
        if found:
            self._class_names = found
        return self._class_names


    def _get_train_split(self) -> str:
        if "train" in self.hf_dataset:
            return "train"
        keys = list(self.hf_dataset.keys())
        return str(keys[0]) if keys else "train"

    def _get_eval_split(self) -> str:
        for name in ("validation", "val", "test", "dev"):
            if name in self.hf_dataset:
                return name
        return self._get_train_split()

    @property
    def train_split_name(self) -> str:
        return self._get_train_split()

    @property
    def eval_split_name(self) -> str:
        return self._get_eval_split()

    @property
    def splits(self) -> list[str]:
        return [str(k) for k in self.hf_dataset.keys()]

    def sample_images(self, n: int = 100) -> list[Any]:
        """Return up to *n* PIL images from the train split (for benchmarking).

        Finds the image column from ``column_map`` if set, otherwise scans
        features for the first ``Image``-type column.
        """
        from engine.column_mapper import classify_feature

        split = self._get_train_split()
        ds = self.hf_dataset[split]

        # Resolve image column
        image_col = (self.column_map or {}).get("image") or (self.column_map or {}).get(
            "input"
        )
        if image_col is None:
            # Scan features for first Image-type column
            for col_name, feat in ds.features.items():
                if classify_feature(feat) == "image":
                    image_col = col_name
                    break
        if image_col is None:
            logger.warning("No image column found in dataset features")
            return []

        images: list[Any] = []
        for i in range(min(n, len(ds))):
            try:
                img = ds[i][image_col]
                if hasattr(img, "convert"):
                    img = img.convert("RGB")
                images.append(img)
            except Exception:
                continue
        return images


    def for_hf(
        self,
        processor: Any,
        head_category: str,
        column_map: dict[str, str] | None = None,
    ) -> Any:
        """Prepare dataset for HF Trainer. Returns transformed HF dataset."""
        raise NotImplementedError("UnifiedDataset.for_hf() not yet implemented")

    def for_ultralytics(
        self,
        task: str,
        output_dir: Path,
        id2label: dict[int, str],
    ) -> Path:
        """Export to YOLO format. Returns path to data.yaml."""
        raise NotImplementedError(
            "UnifiedDataset.for_ultralytics() not yet implemented"
        )
