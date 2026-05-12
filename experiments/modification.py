"""Research modification module — agent writes this per experiment iteration.

All functions are OPTIONAL.  If absent or returning None, the auto-inference
engine handles everything with defaults.

This file is the code experiment surface for the research loop (Phase 4).
The agent can modify architectures, design novel modules, write custom losses,
and the benchmark harness evaluates everything uniformly.

Works on BOTH HF Transformers and Ultralytics models via the UnifiedModel API.
"""

from __future__ import annotations

from typing import Any


def modify_model(model: Any, config: dict[str, Any]) -> Any | None:
    """Alter the model architecture.

    Parameters
    ----------
    model:
        A ``UnifiedModel`` (``HFModel`` or ``UltralyticsModel``).
        Access the raw PyTorch module via ``model.nn_module``.
        Use ``model.find_module_by_role("classification_head")`` for discovery.
        Use ``model.replace_module(path, new_mod)`` for surgery.
    config:
        Dict with ``head_category``, ``image_size``, ``num_classes``, ``class_names``.

    Returns
    -------
    The modified model, or None to use the model as-is.
    """
    return None


def modify_loss(outputs: Any, labels: Any) -> Any | None:
    """Custom loss function.

    Parameters
    ----------
    outputs:
        Model forward() outputs (HF ModelOutput or raw tensors).
    labels:
        Ground truth labels.

    Returns
    -------
    Loss tensor, or None to use the default loss.
    """
    return None


def modify_data(dataset: Any, model: Any) -> Any | None:
    """Custom data preparation.

    Parameters
    ----------
    dataset:
        A ``UnifiedDataset`` instance.
    model:
        The loaded ``UnifiedModel``.

    Returns
    -------
    Modified dataset, or None for default.
    """
    return None


def modify_metrics(eval_pred: Any) -> dict[str, float] | None:
    """Custom evaluation metrics.

    Parameters
    ----------
    eval_pred:
        An ``EvalPrediction`` namedtuple with ``.predictions`` and ``.label_ids``.

    Returns
    -------
    Dict of metric_name → value, or None for defaults.
    """
    return None


def freeze_strategy(model: Any) -> None:
    """Define which parameters to freeze/unfreeze.

    Parameters
    ----------
    model:
        A ``UnifiedModel``.  Use ``model.freeze_except(paths)`` to freeze
        all parameters except those matching the given path prefixes.

    Examples
    --------
    # Freeze everything except custom head:
    model.freeze_except(["templates", "custom_head"])

    # Or for YOLO numbered layers:
    model.freeze_except(["model.24.templates"])

    # Or for HF DETR:
    model.freeze_except(["decoder.class_embed"])
    """
    pass  # Default: no freezing (all params trainable)


def custom_trainer_class() -> type | None:
    """Return a custom Ultralytics trainer subclass.

    Only used for Ultralytics backend models.  Return None to use the
    default trainer.

    Example
    -------
    from ultralytics.models.yolo.detect import DetectionTrainer
    import torch

    class CustomTrainer(DetectionTrainer):
        def optimizer_step(self):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            self.optimizer.step()
            self.optimizer.zero_grad()

    return CustomTrainer
    """
    return None
