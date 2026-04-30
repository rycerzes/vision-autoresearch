"""Backward-compatible entrypoint for ``detect_yolo``.

Canonical multi-task trainer: [`train_ultralytics.py`](train_ultralytics.py).
"""

from train_ultralytics import main

if __name__ == "__main__":
    main()
