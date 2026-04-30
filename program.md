# Program

This repo is a self-contained vision-autoresearch benchmark checkout with a
local promoted master and a git-tracked run ledger.

Core rules:

- use `uv`
- edit config YAMLs only, never training scripts
- never edit `prepare.py`
- refresh from the current local promoted master before a fresh experiment:
  `uv run scripts/refresh_master.py`
- treat `research/live/master.json` and `research/results.tsv` as the
  benchmark source of truth
- run exactly one managed experiment per hypothesis
- record every completed run with
  `uv run scripts/submit_patch.py --comment "..."`
- promotion is local: `scripts/submit_patch.py` updates the live master
  snapshots only when the observed metric beats current master
- all vision metrics are **higher-is-better** (mAP, accuracy, IoU, Dice)

Primary workflow:

1. `uv sync`
2. `uv run scripts/refresh_master.py`
3. edit config YAML (one knob change)
4. `uv run prepare.py --dataset <name> --task <task> --split train`
5. `uv run scripts/hf_job.py preflight --task <task>`
6. `uv run scripts/hf_job.py launch --task <task> --config <config-path>`
7. `uv run scripts/hf_job.py logs <JOB_ID> --follow --output /tmp/vision-run.log`
8. `uv run scripts/parse_metric.py /tmp/vision-run.log`
9. `uv run scripts/submit_patch.py --comment "..."`

Supported tasks:

- `detect` — object detection (DETR, D-FINE, RT-DETR, YOLOS)
- `classify` — image classification (ViT, timm models)
- `segment` — segmentation (SAM, SAM2)
- `detect_yolo` — Ultralytics YOLO detection

Local execution:

- `uv run scripts/run_local.py --task <task> --config <config-path>`
