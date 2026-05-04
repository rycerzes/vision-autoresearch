# Program

This repo is a self-contained vision-autoresearch benchmark checkout with a
local promoted master and a git-tracked run ledger.

Core rules:

- use `uv`
- for benchmark experiments, edit config YAMLs only — do not change training scripts (`train_*.py`) or lab orchestration as part of a hypothesis
- refresh from the current local promoted master before a fresh experiment:
  `uv run scripts/refresh_master.py`
- treat `research/live/master.json` and `research/results.tsv` as the
  benchmark source of truth
- run exactly one managed experiment per hypothesis
- record every completed run with
  `uv run scripts/submit_patch.py --comment "..."`
- promotion is local: `scripts/submit_patch.py` updates the live master only when the configured **`promotion:`** policy beats the current master (direction follows `scripts/vision_lab/metrics.py`).
Primary workflow:

1. `uv sync`
2. `uv run scripts/refresh_master.py`
3. edit config YAML (one knob change)
4. `uv run prepare.py --dataset <name_or_path> --task <task> --split train` (optional `--adapter`, `--run-output-dir`; manifests default to `.runtime/datasets/` or `<run>/dataset/`)
5. `uv run scripts/hf_job.py preflight --task <task>` (checks task ↔ `task_type`, promotion block, model backend, optional `dataset_adapter` / local `dataset_root`)
6. `uv run scripts/hf_job.py launch --task <task> --config <config-path>`
7. `uv run scripts/hf_job.py logs <JOB_ID> --follow --output /tmp/vision-run.log`
8. `uv run scripts/parse_metric.py /tmp/vision-run.log`
9. `uv run scripts/submit_patch.py --comment "..."`

Supported tasks:

- `detect` — object detection (DETR, D-FINE, RT-DETR, YOLOS)
- `classify` — image classification (ViT, timm models)
- `segment` — segmentation (SAM, SAM2)
- `detect_yolo` — Ultralytics YOLO detection via `train_ultralytics.py` (YOLO-World / RT-DETR via `YOLO()`, YOLOE via bridge; YOLO-NAS is not trainable in Ultralytics)
- `track_yolo` — Ultralytics detector training via `train_ultralytics.py` (tracking uses inference-time trackers)
- `segment_yolo` — Ultralytics YOLO segmentation via `train_ultralytics.py` (HF mask column → YOLO labels)
- `classify_yolo` — Ultralytics YOLO classification via `train_ultralytics.py`
- `pose_yolo` — Ultralytics YOLO pose via `train_ultralytics.py` (requires `objects["keypoints"]` in HF data)
- `obb_yolo` — Ultralytics YOLO oriented boxes via `train_ultralytics.py` (5- or 8-value boxes per instance)

Local execution:

- `uv run scripts/run_local.py --task <task> --config <config-path>` (runs the same preflight as HF Jobs unless `--skip-preflight`)
