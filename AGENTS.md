# Agent Instructions

This repo is a live vision-autoresearch experiment repo with a local promoted
master per task type.

## Goal

Maximize the promotion metric (mAP, accuracy, IoU, or Dice depending on task)
on a given dataset with disciplined, comparable single-change experiments.

## Hard Rules

- Edit config YAMLs only, never training scripts (`train_detect.py`,
  `train_ultralytics.py`, `train_classify.py`, `train_segment.py`).
  (`train_detect_yolo.py` is a thin compatibility shim; do not edit it.)
- Never modify `prepare.py`.
- Start from the current local promoted master config, not stale local history.
- Treat `research/live/master.json`, `research/results.tsv`, and the base
  config YAMLs in `configs/` as the benchmark source of truth.
- Do not use repo git history such as `main` or `origin/main` to decide whether
  an experiment is fresh.
- Make exactly one hypothesis change per run (one config knob).
- Run the benchmark before claiming success.
- Record every completed run with
  `uv run scripts/submit_patch.py --comment "..."`
- Promotion is local and only happens when the promotion metric beats current
  master. All vision metrics are **higher-is-better**.
- Keep machine-local compatibility shims out of promoted configs.

## Supported Tasks

| Task | Training Script | Default Model | Promotion Metric |
|------|----------------|---------------|-----------------|
| `detect` | `train_detect.py` | `ustc-community/dfine-small-coco` | mAP |
| `detect_yolo` | `train_ultralytics.py` (via `train_detect_yolo.py`) | `yolo26n.pt` (Ultralytics) | mAP |
| `track_yolo` | `train_ultralytics.py` | `yolo26n.pt` | mAP (detector training for tracking) |
| `segment_yolo` | `train_ultralytics.py` | `yolo26n-seg.pt` | IoU (mask mAP proxy) |
| `classify_yolo` | `train_ultralytics.py` | `yolo26n-cls.pt` | accuracy |
| `pose_yolo` | `train_ultralytics.py` | `yolo26n-pose.pt` | mAP |
| `obb_yolo` | `train_ultralytics.py` | `yolo26n-obb.pt` | mAP |
| `classify` | `train_classify.py` | `google/vit-base-patch16-224` | accuracy |
| `segment` | `train_segment.py` | `facebook/sam2.1-hiera-small` | IoU |

## Config YAML as Experiment Surface

Unlike the upstream autoresearch (which edits `train.py`), this repo uses
YAML config files as the experiment surface. Each config is parsed by
`HfArgumentParser` in the training scripts. Experiments modify config values
such as:

- `learning_rate`, `weight_decay`, `warmup_steps`, `lr_scheduler_type`
- `per_device_train_batch_size`, `gradient_accumulation_steps`
- `image_size`, `use_albumentations`, `use_trivial_augment`
- `freeze_backbone`, `prompt_type`, `loss_type`
- `num_train_epochs`

For **YOLO-family tasks** (`*_yolo`), Ultralytics is the trainer. Use the
top-level YAML mapping `ultralytics_train` to pass
[Ultralytics `train` settings](https://docs.ultralytics.com/modes/train/)
(`lr0`, `weight_decay`, `warmup_epochs`, `cos_lr`, `mosaic`, `patience`,
`device`, …). Keys omitted there still default from `num_train_epochs` →
`epochs`, `per_device_train_batch_size` → `batch`, `image_square_size` →
`imgsz`, `dataloader_num_workers` → `workers`, `seed`, and `fp16` → `amp`.
Hugging Face-only fields such as `learning_rate`, `lr_scheduler_type`,
`warmup_steps`, and `gradient_accumulation_steps` do not affect YOLO unless you
mirror them under `ultralytics_train`. Keys `data`, `project`, `name`, and
`exist_ok` are always set by `train_ultralytics.py`.

Training scripts are stable infrastructure and should not change between
experiments.

## Agent Roles

- **planner** — plans campaigns and experiments; read-only.
- **reviewer** — reviews experiment results; read-only.
- **experiment-worker** — runs in isolated worktree; edits config YAML only.
- **memory-keeper** — owns durable markdown updates under `research/`.
- **reporter** — Trackio and experiment board summaries; read-only.
- **researcher** — literature scouting and paper-derived hypotheses; read-only.

## Managed Runner

Default benchmark path is Hugging Face Jobs.

Per experiment:
- `uv run scripts/hf_job.py preflight --task <detect|detect_yolo|track_yolo|segment_yolo|classify_yolo|pose_yolo|obb_yolo|classify|segment>`
- `uv run scripts/hf_job.py launch --task <task> --config <config-path>`
- `uv run scripts/hf_job.py logs <JOB_ID> --follow --output /tmp/vision-run.log`
- `uv run scripts/parse_metric.py /tmp/vision-run.log`
- `uv run scripts/submit_patch.py --comment "..."`

## Local Runner

For local GPU execution:
- `uv run scripts/run_local.py --task <task> --config <config-path>`

## Standard Workflow

1. Refresh from the local promoted master:
   - `uv run scripts/refresh_master.py`
2. Edit config YAML (one change only).
3. Validate dataset:
   - `uv run prepare.py --dataset <name> --task <task> --split train`
4. Launch benchmark:
   - `uv run scripts/hf_job.py launch --task <task> --config <config-path>`
   - `uv run scripts/hf_job.py logs <JOB_ID> --follow --output /tmp/vision-run.log`
5. Parse the result:
   - `uv run scripts/parse_metric.py /tmp/vision-run.log`
6. Record the hypothesis and outcome in `research/notes.md`.
7. Record the run locally:
   - `uv run scripts/submit_patch.py --comment "..."`

## Repo Layout

- `configs/` — base and experiment config YAMLs (the experiment surface).
- `train_detect.py`, `train_ultralytics.py`, `train_detect_yolo.py` (shim),
  `train_classify.py`, `train_segment.py` — stable training scripts (do not edit during experiments).
- `prepare.py` — dataset validation (never edit).
- `research/results.tsv` — append-only local run ledger.
- `research/live/` — current local promoted master and DAG.
- `research/reference/` — seed master snapshots.
- `research/notes.md` — experiment notebook.
- `research/do-not-repeat.md` — failed experiment guidance.
- `research/paper-ideas.md` — literature-derived hypotheses.
- `research/templates/` — campaign, experiment, and do-not-repeat templates.
- `research/campaigns/` — active campaign docs.
- `research/experiments/` — experiment docs.
- `scripts/` — orchestration scripts.
- `program.md` — benchmark entrypoint note.

## Literature Scouting

When the task is planner or literature research rather than a benchmark run:

- You may edit `research/*.md` and operator docs.
- Translate papers into single-change config hypotheses.
- Record paper-derived ideas in `research/paper-ideas.md`.
- Do not claim a win without a benchmark run.
- `research/live/` — current local promoted master and DAG.
- `research/reference/` — seed master snapshots.
- `research/templates/` — campaign, experiment, and do-not-repeat templates.
- `research/campaigns/` — active campaign docs.
- `research/experiments/` — experiment docs.
- `scripts/` — orchestration scripts.

## Literature Scouting

When the task is planner or literature research rather than a benchmark run:

- You may edit `research/*.md` and operator docs.
- Translate papers into single-change config hypotheses.
- Record paper-derived ideas in `research/paper-ideas.md`.
- Do not claim a win without a benchmark run.
