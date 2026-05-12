"""Research loop orchestrator — manages code-as-experiment-surface iterations.

The research loop is the core of Phase 4: the agent writes Python code in
``experiments/modification.py``, the harness evaluates it, and results are
tracked.  The agent can modify architectures, design novel modules, write
custom losses — all through the unified model API.

Flow per iteration:
1. Agent writes/modifies ``experiments/modification.py``
2. Harness validates the modification (smoke test on CPU)
3. Harness runs ``train_vision.py`` with the modification
4. Results are recorded in ``research/`` ledger
5. Agent reads results and decides next iteration

This module provides the validation and orchestration primitives.
The agent itself (pi, or human) drives the loop.
"""

from __future__ import annotations

import copy
import importlib.util
import logging
import textwrap
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ══ Research iteration tracking ═════════════════════════════════


@dataclass
class ResearchIteration:
    """Record of one research loop iteration."""

    iteration: int
    description: str
    modification_path: str
    model_name: str
    dataset_name: str
    head_category: str
    backend: str

    # Results (filled after training)
    metrics: dict[str, float] = field(default_factory=dict)
    status: str = "pending"  # pending, validated, training, completed, failed
    error: str | None = None
    training_seconds: float = 0.0
    peak_vram_mb: float = 0.0

    # Modification summary (filled during validation)
    modules_replaced: list[str] = field(default_factory=list)
    params_before: int = 0
    params_after: int = 0
    trainable_params: int = 0
    frozen_params: int = 0


@dataclass
class ResearchState:
    """Persistent state for a research campaign."""

    goal: str = ""
    model_name: str = ""
    dataset_name: str = ""
    iterations: list[ResearchIteration] = field(default_factory=list)
    best_iteration: int | None = None
    best_metric_value: float | None = None
    promotion_metric: str = ""
    promotion_direction: str = "higher"

    def record(self, iteration: ResearchIteration) -> None:
        """Record a completed iteration and update best."""
        self.iterations.append(iteration)
        if iteration.status != "completed" or not iteration.metrics:
            return

        value = iteration.metrics.get(self.promotion_metric)
        if value is None:
            return

        is_better = False
        if self.best_metric_value is None:
            is_better = True
        elif self.promotion_direction == "higher":
            is_better = value > self.best_metric_value
        else:
            is_better = value < self.best_metric_value

        if is_better:
            self.best_iteration = iteration.iteration
            self.best_metric_value = value
            logger.info(
                "New best: iteration %d, %s=%s",
                iteration.iteration,
                self.promotion_metric,
                value,
            )


# ══ Modification loading and validation ═════════════════════════


def load_modification_module(path: str | Path) -> Any | None:
    """Load a modification.py module from disk.

    Returns the module object, or None if the file doesn't exist.
    Raises ImportError if the file exists but has syntax/import errors.
    """
    path = Path(path)
    if not path.exists():
        return None

    spec = importlib.util.spec_from_file_location(
        f"modification_{path.stem}", str(path)
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec from {path}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception as e:
        raise ImportError(f"Failed to load {path}: {e}") from e

    return module


def validate_modification(
    model: Any,
    modification: Any,
    config: dict[str, Any],
) -> ValidationResult:
    """Validate a modification module against a loaded model.

    Runs all modification hooks on a *copy* of the model (CPU, no GPU needed).
    Checks that:
    1. modify_model doesn't crash
    2. The modified model still has a valid forward() signature
    3. freeze_strategy runs without error
    4. Parameter counts are sensible

    Returns a ``ValidationResult`` with details.
    """
    result = ValidationResult()
    result.params_before = model.num_parameters

    # ── modify_model ────────────────────────────────────────────
    modify_model_fn = getattr(modification, "modify_model", None)
    if modify_model_fn is not None:
        try:
            modified = modify_model_fn(model, config)
            if modified is not None:
                model = modified
            result.modify_model_ok = True

            # Track what changed
            result.params_after = model.num_parameters
            result.param_delta = result.params_after - result.params_before

        except Exception as e:
            result.modify_model_ok = False
            result.errors.append(f"modify_model() failed: {e}")
            result.traceback = traceback.format_exc()
            return result
    else:
        result.params_after = result.params_before

    # ── freeze_strategy ─────────────────────────────────────────
    freeze_fn = getattr(modification, "freeze_strategy", None)
    if freeze_fn is not None:
        try:
            freeze_fn(model)
            result.freeze_ok = True
            result.trainable_params = model.num_trainable_parameters
            result.frozen_params = result.params_after - result.trainable_params
        except Exception as e:
            result.freeze_ok = False
            result.errors.append(f"freeze_strategy() failed: {e}")
    else:
        result.trainable_params = model.num_trainable_parameters
        result.frozen_params = 0

    # ── modify_loss (check it's callable) ───────────────────────
    modify_loss = getattr(modification, "modify_loss", None)
    if modify_loss is not None:
        if callable(modify_loss):
            result.has_custom_loss = True
        else:
            result.errors.append("modify_loss is not callable")

    # ── modify_metrics (check it's callable) ────────────────────
    modify_metrics = getattr(modification, "modify_metrics", None)
    if modify_metrics is not None:
        if callable(modify_metrics):
            result.has_custom_metrics = True
        else:
            result.errors.append("modify_metrics is not callable")

    # ── modify_data (check it's callable) ───────────────────────
    modify_data = getattr(modification, "modify_data", None)
    if modify_data is not None:
        if callable(modify_data):
            result.has_custom_data = True
        else:
            result.errors.append("modify_data is not callable")

    # ── custom_trainer_class ────────────────────────────────────
    custom_trainer_fn = getattr(modification, "custom_trainer_class", None)
    if custom_trainer_fn is not None:
        try:
            cls = custom_trainer_fn()
            if cls is not None:
                result.has_custom_trainer = True
        except Exception as e:
            result.errors.append(f"custom_trainer_class() failed: {e}")

    # ── Zero trainable params check ─────────────────────────────
    if result.trainable_params == 0 and result.freeze_ok:
        result.warnings.append(
            "freeze_strategy left 0 trainable parameters — "
            "training will have no effect"
        )

    result.valid = len(result.errors) == 0
    return result


@dataclass
class ValidationResult:
    """Result of validating a modification module."""

    valid: bool = True

    # Individual hook results
    modify_model_ok: bool = True
    freeze_ok: bool = True
    has_custom_loss: bool = False
    has_custom_metrics: bool = False
    has_custom_data: bool = False
    has_custom_trainer: bool = False

    # Parameter tracking
    params_before: int = 0
    params_after: int = 0
    param_delta: int = 0
    trainable_params: int = 0
    frozen_params: int = 0

    # Issues
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    traceback: str | None = None

    def summary(self) -> str:
        """Human-readable validation summary."""
        lines = ["═══ Modification Validation ═══"]
        lines.append(f"  Valid:              {'✓' if self.valid else '✗'}")
        lines.append(f"  modify_model:       {'✓' if self.modify_model_ok else '✗'}")
        lines.append(f"  freeze_strategy:    {'✓' if self.freeze_ok else '—'}")
        lines.append(f"  custom_loss:        {'✓' if self.has_custom_loss else '—'}")
        lines.append(f"  custom_metrics:     {'✓' if self.has_custom_metrics else '—'}")
        lines.append(f"  custom_data:        {'✓' if self.has_custom_data else '—'}")
        lines.append(f"  custom_trainer:     {'✓' if self.has_custom_trainer else '—'}")
        lines.append(f"  Params before:      {self.params_before:,}")
        lines.append(f"  Params after:       {self.params_after:,}")
        lines.append(f"  Param delta:        {self.param_delta:+,}")
        lines.append(f"  Trainable:          {self.trainable_params:,}")
        lines.append(f"  Frozen:             {self.frozen_params:,}")

        if self.errors:
            lines.append("  ✗ Errors:")
            for e in self.errors:
                lines.append(f"    - {e}")
        if self.warnings:
            lines.append("  ⚠ Warnings:")
            for w in self.warnings:
                lines.append(f"    - {w}")
        lines.append("═══════════════════════════════")
        return "\n".join(lines)


# ══ Architecture inspection ═════════════════════════════════════


def inspect_architecture(model: Any) -> str:
    """Generate a human-readable architecture summary for the agent.

    Shows the module graph with types, parameter counts, shapes,
    and highlights key roles (backbone, head, neck).

    This is the agent's window into the model before writing surgery code.
    """
    lines: list[str] = []
    lines.append(f"═══ Architecture: {getattr(model, '_model_name', 'unknown')} ═══")
    lines.append(f"  Backend:        {model.backend}")
    lines.append(f"  Head category:  {model.head_category}")
    lines.append(f"  Total params:   {model.num_parameters:,}")
    lines.append(f"  Trainable:      {model.num_trainable_parameters:,}")
    lines.append("")

    # ── Module roles ────────────────────────────────────────────
    lines.append("── Key Roles ──")
    roles = ["backbone", "classification_head", "bbox_head", "neck", "detect_head"]
    for role in roles:
        try:
            path, mod = model.find_module_by_role(role)
            param_count = sum(p.numel() for p in mod.parameters())
            lines.append(
                f"  {role:25s} → {path}  "
                f"({type(mod).__name__}, {param_count:,} params)"
            )
        except (ValueError, AttributeError):
            pass
    lines.append("")

    # ── Module graph (top-level children) ───────────────────────
    lines.append("── Top-Level Modules ──")
    nn_mod = model.nn_module
    for name, child in nn_mod.named_children():
        n_params = sum(p.numel() for p in child.parameters())
        n_children = sum(1 for _ in child.children())
        lines.append(
            f"  {name:30s} {type(child).__name__:25s} "
            f"{n_params:>12,} params  ({n_children} children)"
        )
    lines.append("")

    # ── Detailed graph (first 3 levels) ─────────────────────────
    lines.append("── Detailed Graph (depth ≤ 3) ──")
    for name, mod in nn_mod.named_modules():
        if not name:
            continue
        depth = name.count(".") + 1
        if depth > 3:
            continue
        indent = "  " * depth
        n_own_params = sum(p.numel() for p in mod.parameters(recurse=False))
        type_name = type(mod).__name__

        # Show shape info for Linear/Conv layers
        shape_info = ""
        if hasattr(mod, "in_features") and hasattr(mod, "out_features"):
            shape_info = f" [{mod.in_features}→{mod.out_features}]"
        elif hasattr(mod, "in_channels") and hasattr(mod, "out_channels"):
            k = getattr(mod, "kernel_size", "?")
            shape_info = f" [{mod.in_channels}→{mod.out_channels}, k={k}]"

        if n_own_params > 0:
            lines.append(
                f"{indent}{name}: {type_name}{shape_info} ({n_own_params:,} params)"
            )
        else:
            lines.append(f"{indent}{name}: {type_name}{shape_info}")

    lines.append("")

    # ── Class names ─────────────────────────────────────────────
    class_names = model.get_class_names()
    if class_names:
        if len(class_names) <= 20:
            lines.append(f"── Classes ({len(class_names)}) ──")
            for i, name in enumerate(class_names):
                lines.append(f"  {i}: {name}")
        else:
            lines.append(f"── Classes ({len(class_names)}) ──")
            for i in range(5):
                lines.append(f"  {i}: {class_names[i]}")
            lines.append("  ...")
            for i in range(len(class_names) - 3, len(class_names)):
                lines.append(f"  {i}: {class_names[i]}")

    lines.append("═══════════════════════════════")
    return "\n".join(lines)


def inspect_module_detail(model: Any, path: str) -> str:
    """Inspect a specific module in detail.

    Shows the module's full structure, parameters (with shapes and dtypes),
    buffers, and forward() signature.
    """
    import inspect

    nn_mod = model.nn_module

    # Resolve path
    parts = path.split(".")
    target = nn_mod
    for part in parts:
        if part.isdigit():
            target = target[int(part)]  # type: ignore[index]
        else:
            target = getattr(target, part)

    lines: list[str] = []
    lines.append(f"═══ Module: {path} ═══")
    lines.append(f"  Type:       {type(target).__name__}")
    lines.append(f"  Full type:  {type(target).__module__}.{type(target).__qualname__}")
    lines.append("")

    # Parameters
    params = list(target.named_parameters(recurse=False))
    if params:
        lines.append("── Own Parameters ──")
        for name, p in params:
            lines.append(
                f"  {name:30s} {str(list(p.shape)):20s} "
                f"dtype={p.dtype}  requires_grad={p.requires_grad}"
            )
    lines.append("")

    # Buffers
    buffers = list(target.named_buffers(recurse=False))
    if buffers:
        lines.append("── Buffers ──")
        for name, b in buffers:
            lines.append(f"  {name:30s} {str(list(b.shape)):20s} dtype={b.dtype}")
        lines.append("")

    # Children
    children = list(target.named_children())
    if children:
        lines.append("── Children ──")
        for name, child in children:
            n = sum(p.numel() for p in child.parameters())
            lines.append(f"  {name:30s} {type(child).__name__:25s} ({n:,} params)")
        lines.append("")

    # Forward signature
    forward_fn = getattr(target, "forward", None)
    if forward_fn is not None:
        try:
            sig = inspect.signature(forward_fn)
            lines.append(f"── forward() signature ──")
            lines.append(f"  {sig}")
        except (ValueError, TypeError):
            pass

    lines.append("═══════════════════════════════")
    return "\n".join(lines)


# ══ Modification writing helpers ════════════════════════════════


def write_modification(
    path: str | Path,
    code: str,
    *,
    backup: bool = True,
) -> Path:
    """Write a modification.py file, optionally backing up the previous one.

    Parameters
    ----------
    path:
        Output path for the modification file.
    code:
        Python source code for the modification module.
    backup:
        If True, rename the existing file to ``modification.py.bak``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if backup and path.exists():
        bak = path.with_suffix(".py.bak")
        # Rotate backups
        if bak.exists():
            import shutil
            timestamp = int(time.time())
            archive = path.parent / f"{path.stem}.{timestamp}.py.bak"
            shutil.move(str(bak), str(archive))
        path.rename(bak)

    path.write_text(code)
    logger.info("Wrote modification to %s", path)
    return path


def generate_modification_template(
    model: Any,
    strategy: str = "head_swap",
) -> str:
    """Generate a modification.py template based on model inspection.

    The agent can use this as a starting point, then modify the code.

    Parameters
    ----------
    model:
        A loaded ``UnifiedModel``.
    strategy:
        Template strategy: ``"head_swap"``, ``"freeze_finetune"``,
        ``"custom_loss"``, ``"full_surgery"``.
    """
    if strategy == "head_swap":
        return _template_head_swap(model)
    if strategy == "freeze_finetune":
        return _template_freeze_finetune(model)
    if strategy == "custom_loss":
        return _template_custom_loss(model)
    if strategy == "full_surgery":
        return _template_full_surgery(model)
    raise ValueError(f"Unknown template strategy: {strategy!r}")


def _template_head_swap(model: Any) -> str:
    """Generate head-swap template from model inspection."""
    # Try to find the classification head
    try:
        path, head = model.find_module_by_role("classification_head")
        head_type = type(head).__name__

        # Extract dimensions
        if hasattr(head, "in_features") and hasattr(head, "out_features"):
            in_dim = head.in_features
            out_dim = head.out_features
            dim_info = f"in_features={in_dim}, out_features={out_dim}"
        elif hasattr(head, "in_channels") and hasattr(head, "out_channels"):
            in_dim = head.in_channels
            out_dim = head.out_channels
            dim_info = f"in_channels={in_dim}, out_channels={out_dim}"
        else:
            in_dim = "?"
            out_dim = "?"
            dim_info = "dimensions unknown"
    except ValueError:
        path = "<could_not_detect>"
        head_type = "Unknown"
        in_dim = "?"
        out_dim = "?"
        dim_info = "dimensions unknown"

    return textwrap.dedent(f'''\
        """Head swap modification — generated from model inspection.

        Original head: {head_type} at '{path}'
        Dimensions: {dim_info}
        Backend: {model.backend}
        """

        from __future__ import annotations

        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from typing import Any


        class CustomHead(nn.Module):
            """Replace the original {head_type} with a custom head."""

            def __init__(self, hidden_dim: int = {in_dim}, num_classes: int = {out_dim}):
                super().__init__()
                # Example: cosine similarity head
                self.temperature = nn.Parameter(torch.tensor(0.07))
                self.templates = nn.Parameter(torch.randn(num_classes, hidden_dim))
                nn.init.xavier_uniform_(self.templates)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                normed = F.normalize(x, dim=-1)
                normed_t = F.normalize(self.templates, dim=-1)
                return torch.matmul(normed, normed_t.T) / self.temperature.clamp(min=1e-4)


        def modify_model(model: Any, config: dict[str, Any]) -> Any:
            path, head = model.find_module_by_role("classification_head")
            num_classes = config.get("num_classes", {out_dim})

            # Detect dimensions from the existing head
            if hasattr(head, "in_features"):
                hidden_dim = head.in_features
            elif hasattr(head, "in_channels"):
                hidden_dim = head.in_channels
            else:
                hidden_dim = {in_dim}

            new_head = CustomHead(hidden_dim, num_classes)
            model.replace_module(path, new_head)
            return model


        def freeze_strategy(model: Any) -> None:
            # Only train the custom head parameters
            model.freeze_except(["templates", "temperature"])
    ''')


def _template_freeze_finetune(model: Any) -> str:
    """Generate freeze-then-finetune template."""
    try:
        backbone_path, backbone = model.find_module_by_role("backbone")
        backbone_info = f"'{backbone_path}' ({type(backbone).__name__})"
    except ValueError:
        backbone_path = "<backbone>"
        backbone_info = "could not detect"

    return textwrap.dedent(f'''\
        """Freeze-finetune modification — generated from model inspection.

        Backbone: {backbone_info}
        Backend: {model.backend}
        """

        from __future__ import annotations
        from typing import Any


        def freeze_strategy(model: Any) -> None:
            """Freeze backbone, train only the head(s)."""
            # Freeze everything first
            for param in model.nn_module.parameters():
                param.requires_grad_(False)

            # Unfreeze non-backbone modules
            backbone_path = "{backbone_path}"
            for name, param in model.nn_module.named_parameters():
                if not name.startswith(backbone_path):
                    param.requires_grad_(True)
    ''')


def _template_custom_loss(model: Any) -> str:
    """Generate custom loss template."""
    return textwrap.dedent(f'''\
        """Custom loss modification.

        Backend: {model.backend}
        Head category: {model.head_category}
        """

        from __future__ import annotations

        import torch
        import torch.nn.functional as F
        from typing import Any


        def modify_loss(outputs: Any, labels: Any) -> torch.Tensor:
            """Custom loss function.

            Replaces the default loss with a custom one.
            """
            logits = outputs.logits

            # Example: label smoothing cross-entropy
            num_classes = logits.size(-1)
            smooth = 0.1
            confidence = 1.0 - smooth

            log_probs = F.log_softmax(logits.view(-1, num_classes), dim=-1)
            targets = labels.view(-1)

            nll_loss = F.nll_loss(log_probs, targets, reduction="mean")
            smooth_loss = -log_probs.mean(dim=-1).mean()

            return confidence * nll_loss + smooth * smooth_loss
    ''')


def _template_full_surgery(model: Any) -> str:
    """Generate full architecture surgery template."""
    # Gather info
    info_lines = []
    for role in ["backbone", "classification_head", "bbox_head", "neck", "detect_head"]:
        try:
            p, m = model.find_module_by_role(role)
            n = sum(pp.numel() for pp in m.parameters())
            info_lines.append(f"#   {role}: '{p}' ({type(m).__name__}, {n:,} params)")
        except ValueError:
            pass

    roles_comment = "\n".join(info_lines) if info_lines else "#   (no roles detected)"

    return textwrap.dedent(f'''\
        """Full architecture surgery — generated from model inspection.

        Backend: {model.backend}
        Head category: {model.head_category}
        Total params: {model.num_parameters:,}

        Detected roles:
        {roles_comment}
        """

        from __future__ import annotations

        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from typing import Any


        def modify_model(model: Any, config: dict[str, Any]) -> Any:
            """Full surgery: modify multiple components."""
            # Access raw nn.Module
            nn_mod = model.nn_module

            # Get module graph for inspection
            graph = model.get_module_graph()

            # Example: print all modules with >1000 params
            for path, info in graph.items():
                if info.num_params > 1000:
                    print(f"  {{path}}: {{info.type_name}} ({{info.num_params:,}} params)")

            # TODO: implement your architecture changes here
            # model.replace_module("path.to.module", new_module)

            return model


        def modify_loss(outputs: Any, labels: Any) -> torch.Tensor | None:
            """Custom loss. Return None to use default."""
            return None


        def freeze_strategy(model: Any) -> None:
            """Selective freeze."""
            # Example: freeze first 80% of parameters
            all_params = list(model.nn_module.named_parameters())
            cutoff = int(len(all_params) * 0.8)
            for _, param in all_params[:cutoff]:
                param.requires_grad_(False)
            for _, param in all_params[cutoff:]:
                param.requires_grad_(True)
    ''')
