"""Unified model protocol — the single interface for all vision models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch.nn as nn


@dataclass(frozen=True)
class ModuleInfo:
    """Lightweight summary of one ``nn.Module`` node."""

    type_name: str
    num_params: int
    trainable_params: int

    @classmethod
    def from_module(cls, mod: nn.Module) -> ModuleInfo:
        total = sum(p.numel() for p in mod.parameters(recurse=False))
        trainable = sum(
            p.numel() for p in mod.parameters(recurse=False) if p.requires_grad
        )
        return cls(
            type_name=type(mod).__name__,
            num_params=total,
            trainable_params=trainable,
        )


@runtime_checkable
class UnifiedModel(Protocol):
    """Common interface for all vision models regardless of backend."""


    @property
    def backend(self) -> str:
        """``"hf"`` or ``"ultralytics"``."""
        ...

    @property
    def head_category(self) -> str:
        """High-level task family (``detection``, ``classification``, …)."""
        ...

    @property
    def nn_module(self) -> nn.Module:
        """Raw PyTorch module (for surgery, parameter access, etc.)."""
        ...

    @property
    def num_parameters(self) -> int:
        ...

    @property
    def num_trainable_parameters(self) -> int:
        ...


    def train(self, train_dataset: Any, eval_dataset: Any, args: dict[str, Any]) -> dict[str, Any]:
        """Train the model. Returns metrics dict."""
        ...

    def evaluate(self, dataset: Any) -> dict[str, Any]:
        """Run evaluation. Returns metrics dict with standard keys."""
        ...

    def predict(self, image: Any) -> Any:
        """Single-image inference."""
        ...


    def benchmark_latency(
        self,
        sample_images: list[Any],
        *,
        num_warmup: int = 10,
        num_runs: int = 100,
    ) -> dict[str, float]:
        """Standardised latency measurement.

        Returns at least ``inference_ms``, ``throughput_img_per_sec``,
        ``peak_vram_mb``.
        """
        ...

    def export(self, fmt: str, output_path: Path) -> Path:
        """Export to ONNX / TensorRT / CoreML / …"""
        ...


    def get_module_graph(self) -> dict[str, ModuleInfo]:
        """Map of ``dotted.path`` → :class:`ModuleInfo`."""
        ...

    def find_module_by_role(self, role: str) -> tuple[str, nn.Module]:
        """Locate a module by functional role (``classification_head``,
        ``backbone``, ``bbox_head``, ``neck``, …).
        """
        ...

    def replace_module(self, path: str, new_module: nn.Module) -> None:
        """Swap a module at the given dotted path."""
        ...

    def freeze_except(self, module_paths: list[str]) -> None:
        """Freeze all parameters **except** those under *module_paths*."""
        ...

    def get_class_names(self) -> list[str] | None:
        """Return class names when the model has a fixed vocabulary."""
        ...
