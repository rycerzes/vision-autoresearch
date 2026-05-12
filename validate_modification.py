"""Validate a modification.py against a model before training.

Usage:
    uv run validate_modification.py <model_name> [--modification <path>]

Examples:
    # Validate the default experiments/modification.py
    uv run validate_modification.py ustc-community/dfine-small-coco

    # Validate a custom modification
    uv run validate_modification.py yolo11n.pt --modification experiments/my_mod.py

Runs all modification hooks on a copy of the model (CPU, no GPU needed).
Reports parameter changes, freeze state, and any errors.
"""

from __future__ import annotations

import argparse
import logging
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a modification.py against a model",
    )
    parser.add_argument(
        "model_name",
        help="Model identifier (HF Hub id, .pt file, or local path)",
    )
    parser.add_argument(
        "--modification",
        default="experiments/modification.py",
        help="Path to modification.py (default: experiments/modification.py)",
    )
    parser.add_argument(
        "--head-category",
        metavar="CATEGORY",
        help="Override head category detection",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=logging.WARNING,
    )

    # Load model
    from engine.backend import load_model

    print(f"Loading {args.model_name}...", file=sys.stderr)
    model = load_model(
        args.model_name,
        mode="train",
        head_category_override=args.head_category,
    )
    print(f"Loaded: {model.backend} / {model.head_category}", file=sys.stderr)

    # Load modification
    from engine.research import load_modification_module, validate_modification

    print(f"Loading {args.modification}...", file=sys.stderr)
    modification = load_modification_module(args.modification)
    if modification is None:
        print(f"ERROR: {args.modification} not found", file=sys.stderr)
        sys.exit(1)

    # Build config dict
    config = {
        "head_category": model.head_category,
        "image_size": (640, 640),
        "num_classes": None,
        "class_names": None,
    }

    # Get class info from model
    names = model.get_class_names()
    if names:
        config["num_classes"] = len(names)
        config["class_names"] = names
    else:
        model_config = getattr(model, "config", None)
        if model_config:
            config["num_classes"] = getattr(model_config, "num_labels", None)

    # Validate
    print(f"\nValidating modification...\n", file=sys.stderr)
    result = validate_modification(model, modification, config)

    # Output
    print(result.summary())

    if result.traceback:
        print(f"\n── Traceback ──\n{result.traceback}")

    sys.exit(0 if result.valid else 1)


if __name__ == "__main__":
    main()
