"""Unified dataset abstraction — wraps HF datasets and exports to any format.

Works with both HF Trainer and Ultralytics by converting to each backend's
expected format on demand.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np

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
        self.column_map: dict[str, Any] | None = None
        self._class_names: list[str] | None = None
        logger.info(
            "UnifiedDataset loaded: %s  splits=%s",
            dataset_name,
            list(self.hf_dataset.keys()),
        )


    def auto_map_columns(
        self, head_category: str, processor: Any = None
    ) -> dict[str, Any]:
        """Type-based column alignment. Stores result in ``self.column_map``."""
        from engine.column_mapper import auto_map_columns

        train_split = self._get_train_split()
        features = self.hf_dataset[train_split].features
        self.column_map = auto_map_columns(
            dict(features), head_category, processor=processor
        )
        return self.column_map

    def set_column_map(self, column_map: dict[str, Any]) -> None:
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

    def ensure_train_val_split(self, val_fraction: float = 0.15, seed: int = 42) -> None:
        """Create a validation split if one doesn't already exist.

        Splits the train set into train + validation using ``val_fraction``.
        No-op if a validation/test split already exists.
        """
        for name in ("validation", "val", "test", "dev"):
            if name in self.hf_dataset:
                return

        train_split = self._get_train_split()
        if train_split not in self.hf_dataset:
            return

        logger.info(
            "No validation split found — splitting %.0f%% from train (seed=%d)",
            val_fraction * 100,
            seed,
        )
        split_result = self.hf_dataset[train_split].train_test_split(
            test_size=val_fraction, seed=seed
        )
        self.hf_dataset[train_split] = split_result["train"]
        self.hf_dataset["validation"] = split_result["test"]
        logger.info(
            "Split: train=%d, validation=%d",
            len(self.hf_dataset[train_split]),
            len(self.hf_dataset["validation"]),
        )

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
        *,
        train_augmentation: Any | None = None,
        eval_augmentation: Any | None = None,
        image_size: tuple[int, int] = (224, 224),
    ) -> dict[str, Any]:
        """Prepare dataset for HF Trainer.

        Returns a dict ``{"train": Dataset, "eval": Dataset}`` with columns
        transformed for the processor.  Each example is a dict with
        ``pixel_values``, ``labels``, and any other keys the model expects.

        Parameters
        ----------
        processor:
            HF processor / image processor for encoding images.
        head_category:
            From ``model.head_category``.
        column_map:
            Explicit override.  Falls back to ``self.column_map``.
        train_augmentation:
            Callable from ``engine.augmentation`` for training transforms.
        eval_augmentation:
            Callable from ``engine.augmentation`` for eval transforms.
        image_size:
            Target (H, W) for resize if augmentation not provided.
        """
        cmap = column_map or self.column_map
        if cmap is None:
            raise ValueError(
                "Column map not set. Call auto_map_columns() or provide column_map."
            )

        image_col: str = cmap["image"]
        target_col: str | None = cmap.get("target")
        subfield_map: dict[str, str] | None = cmap.get("target_subfields")  # type: ignore[assignment]  # may be None

        train_split = self._get_train_split()
        eval_split = self._get_eval_split()

        def make_transform(augmentation: Any | None, is_train: bool):
            """Return a map function that transforms one example."""

            def transform_example(example: dict[str, Any]) -> dict[str, Any]:
                img = example[image_col]
                if hasattr(img, "convert"):
                    img = img.convert("RGB")

                # Apply augmentation if provided
                if augmentation is not None:
                    img_array = np.array(img)
                    if head_category in ("detection", "structured_detection"):
                        # Spatial-aligned: pass bboxes/labels
                        aug_kwargs = _prepare_detection_aug_kwargs(
                            example, target_col, img_array, subfield_map
                        )
                        aug_result = augmentation(**aug_kwargs)
                        img_array = aug_result["image"]
                        # Update target with transformed bboxes
                        if target_col and "bboxes" in aug_result:
                            example = dict(example)  # shallow copy
                            example[target_col] = _rebuild_detection_target(
                                example.get(target_col),
                                aug_result["bboxes"],
                                aug_result.get("bbox_labels", []),
                                subfield_map,
                            )
                    elif head_category in ("dense_classification", "prompted_segmentation"):
                        mask_array = None
                        if target_col and target_col in example:
                            t = example[target_col]
                            mask_array = np.array(t) if hasattr(t, "__array__") else t
                        aug_result = augmentation(
                            image=img_array, mask=mask_array
                        )
                        img_array = aug_result["image"]
                        if mask_array is not None and "mask" in aug_result and target_col is not None:
                            example = dict(example)
                            example[target_col] = aug_result["mask"]
                    elif head_category in ("image_reconstruction", "dense_regression", "pair_matching"):
                        target_img = None
                        if target_col and target_col in example:
                            t = example[target_col]
                            target_img = np.array(t) if hasattr(t, "__array__") else t
                        aug_result = augmentation(
                            image=img_array, target_image=target_img
                        )
                        img_array = aug_result["image"]
                        if target_img is not None and "target_image" in aug_result and target_col is not None:
                            example = dict(example)
                            example[target_col] = aug_result["target_image"]
                    else:
                        aug_result = augmentation(image=img_array)
                        img_array = aug_result["image"]

                    # Convert back to PIL for processor
                    from PIL import Image as PILImage
                    img = PILImage.fromarray(img_array)

                # Run through processor
                processed = processor(images=img, return_tensors="pt")
                # Flatten batch dimension (processor adds it)
                result: dict[str, Any] = {}
                for k, v in processed.items():
                    if hasattr(v, "squeeze"):
                        result[k] = v.squeeze(0)
                    else:
                        result[k] = v

                # Add labels
                if target_col and target_col in example:
                    result["labels"] = _format_labels(
                        example[target_col], head_category, subfield_map
                    )

                return result

            return transform_example

        train_ds = self.hf_dataset[train_split].map(
            make_transform(train_augmentation, is_train=True),
            remove_columns=self.hf_dataset[train_split].column_names,
        )

        eval_ds = self.hf_dataset[eval_split].map(
            make_transform(eval_augmentation, is_train=False),
            remove_columns=self.hf_dataset[eval_split].column_names,
        )

        logger.info(
            "for_hf: train=%d examples, eval=%d examples",
            len(train_ds),
            len(eval_ds),
        )
        return {"train": train_ds, "eval": eval_ds}

    def for_ultralytics(
        self,
        task: str,
        output_dir: Path,
        id2label: dict[int, str],
    ) -> Path:
        """Export HF dataset to YOLO format. Returns path to ``data.yaml``.

        Creates the standard Ultralytics directory structure:
        ``images/train/``, ``images/val/``, ``labels/train/``, ``labels/val/``
        and a ``data.yaml`` pointing to them.

        Parameters
        ----------
        task:
            Ultralytics task (``"detect"``, ``"segment"``, ``"classify"``,
            ``"pose"``, ``"obb"``).
        output_dir:
            Root directory for the exported dataset.
        id2label:
            ``{class_id: class_name}`` mapping.
        """
        cmap = self.column_map
        if cmap is None:
            raise ValueError(
                "Column map not set. Call auto_map_columns() before for_ultralytics()."
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if task == "classify":
            return self._export_classification_yolo(output_dir, id2label)
        elif task in ("detect", "obb", "pose"):
            return self._export_detection_yolo(output_dir, id2label, task)
        elif task == "segment":
            return self._export_segmentation_yolo(output_dir, id2label)
        else:
            raise ValueError(f"Unsupported Ultralytics task for export: {task!r}")


    def _export_detection_yolo(
        self,
        output_dir: Path,
        id2label: dict[int, str],
        task: str = "detect",
    ) -> Path:
        """Export detection dataset to YOLO format (images/ + labels/ + data.yaml)."""
        cmap = self.column_map  # guaranteed set by for_ultralytics
        assert cmap is not None
        image_col: str = cmap["image"]
        target_col: str = cmap["target"]
        subfield_map: dict[str, str] | None = cmap.get("target_subfields")

        split_map = {
            self._get_train_split(): "train",
            self._get_eval_split(): "val",
        }

        for hf_split, yolo_split in split_map.items():
            if hf_split not in self.hf_dataset:
                continue

            img_dir = output_dir / "images" / yolo_split
            lbl_dir = output_dir / "labels" / yolo_split
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            ds = self.hf_dataset[hf_split]
            for idx in range(len(ds)):
                example = ds[idx]
                img = example[image_col]
                if hasattr(img, "convert"):
                    img = img.convert("RGB")

                img_path = img_dir / f"{idx:06d}.jpg"
                img.save(str(img_path))

                target = example.get(target_col, {})
                label_lines = _detection_target_to_yolo_lines(
                    target, img.width, img.height, subfield_map, task
                )
                lbl_path = lbl_dir / f"{idx:06d}.txt"
                lbl_path.write_text("\n".join(label_lines))

        # Write data.yaml
        names = {int(k): v for k, v in id2label.items()}
        data_yaml = {
            "path": str(output_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": names,
            "nc": len(names),
        }
        yaml_path = output_dir / "data.yaml"
        import yaml
        yaml_path.write_text(yaml.dump(data_yaml, default_flow_style=False))

        logger.info(
            "Exported detection dataset to YOLO format: %s (%d classes)",
            yaml_path,
            len(names),
        )
        return yaml_path

    def _export_segmentation_yolo(
        self,
        output_dir: Path,
        id2label: dict[int, str],
    ) -> Path:
        """Export instance segmentation dataset to YOLO format.

        YOLO segmentation labels: ``class_id x1 y1 x2 y2 ... xn yn``
        (normalized polygon coordinates).
        """
        cmap = self.column_map
        assert cmap is not None
        image_col: str = cmap["image"]
        target_col: str = cmap["target"]
        subfield_map: dict[str, str] | None = cmap.get("target_subfields")

        split_map = {
            self._get_train_split(): "train",
            self._get_eval_split(): "val",
        }

        for hf_split, yolo_split in split_map.items():
            if hf_split not in self.hf_dataset:
                continue

            img_dir = output_dir / "images" / yolo_split
            lbl_dir = output_dir / "labels" / yolo_split
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            ds = self.hf_dataset[hf_split]
            for idx in range(len(ds)):
                example = ds[idx]
                img = example[image_col]
                if hasattr(img, "convert"):
                    img = img.convert("RGB")

                img_path = img_dir / f"{idx:06d}.jpg"
                img.save(str(img_path))

                target = example.get(target_col, {})
                label_lines = _segmentation_target_to_yolo_lines(
                    target, img.width, img.height, subfield_map
                )
                lbl_path = lbl_dir / f"{idx:06d}.txt"
                lbl_path.write_text("\n".join(label_lines))

        # Write data.yaml
        names = {int(k): v for k, v in id2label.items()}
        data_yaml = {
            "path": str(output_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": names,
            "nc": len(names),
        }
        yaml_path = output_dir / "data.yaml"
        import yaml
        yaml_path.write_text(yaml.dump(data_yaml, default_flow_style=False))

        logger.info(
            "Exported segmentation dataset to YOLO format: %s",
            yaml_path,
        )
        return yaml_path

    def _export_classification_yolo(
        self,
        output_dir: Path,
        id2label: dict[int, str],
    ) -> Path:
        """Export classification dataset to YOLO format.

        YOLO classification layout: ``train/class_name/image.jpg``
        """
        cmap = self.column_map
        assert cmap is not None
        image_col: str = cmap["image"]
        target_col: str = cmap["target"]

        split_map = {
            self._get_train_split(): "train",
            self._get_eval_split(): "val",
        }

        for hf_split, yolo_split in split_map.items():
            if hf_split not in self.hf_dataset:
                continue

            ds = self.hf_dataset[hf_split]
            for idx in range(len(ds)):
                example = ds[idx]
                img = example[image_col]
                if hasattr(img, "convert"):
                    img = img.convert("RGB")

                label = example.get(target_col, 0)
                class_name = id2label.get(int(label), str(label))
                class_dir = output_dir / yolo_split / class_name
                class_dir.mkdir(parents=True, exist_ok=True)

                img_path = class_dir / f"{idx:06d}.jpg"
                img.save(str(img_path))

        # Write data.yaml
        names = {int(k): v for k, v in id2label.items()}
        data_yaml = {
            "path": str(output_dir.resolve()),
            "train": "train",
            "val": "val",
            "names": names,
            "nc": len(names),
        }
        yaml_path = output_dir / "data.yaml"
        import yaml
        yaml_path.write_text(yaml.dump(data_yaml, default_flow_style=False))

        logger.info(
            "Exported classification dataset to YOLO format: %s (%d classes)",
            yaml_path,
            len(names),
        )
        return yaml_path




def _get_subfield(target: dict, role: str, subfield_map: dict[str, str] | None) -> Any | None:
    """Look up a sub-field from a target dict using the type-derived subfield map.

    The ``subfield_map`` maps roles (``"bbox"``, ``"category"``, ``"segmentation"``)
    to the actual field names discovered by ``auto_map_target_subfields``.
    No hardcoded field names — this is the only lookup mechanism.
    """
    if subfield_map is None or role not in subfield_map:
        return None
    actual_key = subfield_map[role]
    return target.get(actual_key)


def _set_subfield(
    target: dict, role: str, value: Any, subfield_map: dict[str, str] | None
) -> None:
    """Write a value back to a target dict using the subfield map."""
    if subfield_map is None or role not in subfield_map:
        return
    actual_key = subfield_map[role]
    target[actual_key] = value


def _format_labels(
    target: Any,
    head_category: str,
    subfield_map: dict[str, str] | None = None,
) -> Any:
    """Convert dataset target to the format expected by the model.

    Uses ``subfield_map`` (from type-based discovery) to find bbox/category
    fields inside structured dicts.  No hardcoded field names.
    """
    import torch

    if head_category == "classification":
        if isinstance(target, (int, float)):
            return torch.tensor(int(target))
        return target

    if head_category in ("detection", "structured_detection"):
        if isinstance(target, dict):
            result: dict[str, Any] = {}
            bboxes = _get_subfield(target, "bbox", subfield_map)
            if bboxes is not None:
                result["boxes"] = (
                    torch.tensor(bboxes, dtype=torch.float32)
                    if not isinstance(bboxes, torch.Tensor)
                    else bboxes
                )
            cats = _get_subfield(target, "category", subfield_map)
            if cats is not None:
                result["class_labels"] = (
                    torch.tensor(cats, dtype=torch.long)
                    if not isinstance(cats, torch.Tensor)
                    else cats
                )
            return result
        return target

    if head_category in ("dense_classification", "prompted_segmentation"):
        if isinstance(target, np.ndarray):
            return torch.tensor(target, dtype=torch.long)
        if hasattr(target, "__array__"):
            return torch.tensor(np.array(target), dtype=torch.long)
        return target

    if head_category == "dense_regression":
        if isinstance(target, np.ndarray):
            return torch.tensor(target, dtype=torch.float32)
        if hasattr(target, "__array__"):
            return torch.tensor(np.array(target), dtype=torch.float32)
        return target

    # Default: return as-is
    return target


def _prepare_detection_aug_kwargs(
    example: dict[str, Any],
    target_col: str | None,
    img_array: Any,
    subfield_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Extract bboxes and labels from a detection example for augmentation.

    Uses ``subfield_map`` to find the right fields.  No hardcoded field names.
    """
    kwargs: dict[str, Any] = {"image": img_array}
    if target_col is None or target_col not in example:
        return kwargs

    target = example[target_col]
    if not isinstance(target, dict):
        return kwargs

    bboxes = _get_subfield(target, "bbox", subfield_map)
    if bboxes is not None:
        kwargs["bboxes"] = (
            bboxes.tolist() if hasattr(bboxes, "tolist") else list(bboxes)
        )

    labels = _get_subfield(target, "category", subfield_map)
    if labels is not None:
        kwargs["bbox_labels"] = (
            labels.tolist() if hasattr(labels, "tolist") else list(labels)
        )
    elif bboxes is not None:
        kwargs["bbox_labels"] = [0] * len(kwargs["bboxes"])

    return kwargs


def _rebuild_detection_target(
    original_target: Any,
    transformed_bboxes: list[Any],
    transformed_labels: list[Any],
    subfield_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Rebuild detection target dict after augmentation.

    Writes back to the same fields identified by ``subfield_map``.
    """
    result: dict[str, Any] = {}
    if isinstance(original_target, dict):
        result = dict(original_target)

    _set_subfield(result, "bbox", transformed_bboxes, subfield_map)
    _set_subfield(result, "category", transformed_labels, subfield_map)

    return result


def _detection_target_to_yolo_lines(
    target: Any,
    img_w: int,
    img_h: int,
    subfield_map: dict[str, str] | None = None,
    task: str = "detect",
) -> list[str]:
    """Convert detection target to YOLO label lines.

    YOLO format: ``class_id cx cy w h`` (normalized).
    Uses ``subfield_map`` to find bbox/category fields.
    """
    lines: list[str] = []
    if not isinstance(target, dict):
        return lines

    bboxes = _get_subfield(target, "bbox", subfield_map)
    if bboxes is None:
        return lines

    labels = _get_subfield(target, "category", subfield_map)
    if labels is None:
        labels = [0] * len(bboxes)

    for bbox, label in zip(bboxes, labels):
        if len(bbox) < 4:
            continue
        # Assume COCO format: [x, y, w, h]
        x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        # Convert to YOLO format: center_x, center_y, width, height (normalized)
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h
        # Clamp to [0, 1]
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        nw = max(0.0, min(1.0, nw))
        nh = max(0.0, min(1.0, nh))
        if nw <= 0 or nh <= 0:
            continue
        lines.append(f"{int(label)} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    return lines


def _segmentation_target_to_yolo_lines(
    target: Any,
    img_w: int,
    img_h: int,
    subfield_map: dict[str, str] | None = None,
) -> list[str]:
    """Convert segmentation target to YOLO polygon lines.

    YOLO segmentation: ``class_id x1 y1 x2 y2 ... xn yn`` (normalized).
    Falls back to bbox if no polygon data available.
    Uses ``subfield_map`` to find segmentation/category fields.
    """
    lines: list[str] = []
    if not isinstance(target, dict):
        return lines

    polygons = _get_subfield(target, "segmentation", subfield_map)
    labels = _get_subfield(target, "category", subfield_map)

    if polygons is not None and labels is not None:
        for poly, label in zip(polygons, labels):
            if isinstance(poly, (list, tuple)) and len(poly) >= 6:
                normalized = []
                for i in range(0, len(poly), 2):
                    if i + 1 < len(poly):
                        nx = float(poly[i]) / img_w
                        ny = float(poly[i + 1]) / img_h
                        normalized.extend([f"{nx:.6f}", f"{ny:.6f}"])
                coords = " ".join(normalized)
                lines.append(f"{int(label)} {coords}")
    else:
        # Fall back to bbox-based lines
        lines = _detection_target_to_yolo_lines(target, img_w, img_h, subfield_map, "detect")

    return lines
