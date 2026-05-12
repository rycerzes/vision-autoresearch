"""Architecture inspection CLI — examine any model before writing surgery code.

Usage:
    uv run inspect_model.py <model_name> [--detail <module_path>] [--template <strategy>]

Examples:
    # Full architecture overview
    uv run inspect_model.py ustc-community/dfine-small-coco

    # YOLO model
    uv run inspect_model.py yolo11n.pt

    # Inspect a specific module in detail
    uv run inspect_model.py ustc-community/dfine-small-coco --detail model.decoder.class_embed

    # Generate a modification template
    uv run inspect_model.py ustc-community/dfine-small-coco --template head_swap
    uv run inspect_model.py yolo11n.pt --template freeze_finetune

The output shows the model's architecture, key roles (backbone, head, neck),
parameter counts, and module shapes — everything the agent needs to write
``experiments/modification.py``.
"""

from __future__ import annotations

import argparse
import logging
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect any vision model architecture",
    )
    parser.add_argument(
        "model_name",
        help="Model identifier (HF Hub id, .pt file, or local path)",
    )
    parser.add_argument(
        "--detail",
        metavar="MODULE_PATH",
        help="Inspect a specific module in detail (dotted path)",
    )
    parser.add_argument(
        "--template",
        choices=["head_swap", "freeze_finetune", "custom_loss", "full_surgery"],
        help="Generate a modification.py template",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="Write template to file (default: stdout)",
    )
    parser.add_argument(
        "--head-category",
        metavar="CATEGORY",
        help="Override head category detection",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict"],
        default="train",
        help="Model loading mode (affects SAM routing)",
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
        mode=args.mode,
        head_category_override=args.head_category,
    )
    print(f"Loaded: {model.backend} / {model.head_category}\n", file=sys.stderr)

    if args.detail:
        # Detailed module inspection
        from engine.research import inspect_module_detail

        print(inspect_module_detail(model, args.detail))

    elif args.template:
        # Generate modification template
        from engine.research import generate_modification_template

        code = generate_modification_template(model, strategy=args.template)

        if args.output:
            from pathlib import Path

            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(code)
            print(f"Template written to {args.output}", file=sys.stderr)
        else:
            print(code)

    else:
        # Full architecture overview
        from engine.research import inspect_architecture

        print(inspect_architecture(model))


if __name__ == "__main__":
    main()
