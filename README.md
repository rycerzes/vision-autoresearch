# Vision Autoresearch

Autonomous vision model fine-tuning research swarm. Runs disciplined,
single-change experiments on object detection, image classification, and
segmentation models using Hugging Face Jobs for execution.

## Supported Tasks

| Task | Script | Default Model | Metric |
|------|--------|---------------|--------|
| Object Detection | `train_detect.py` | `ustc-community/dfine-small-coco` | mAP |
| Image Classification | `train_classify.py` | `google/vit-base-patch16-224` | accuracy |
| Segmentation | `train_segment.py` | `facebook/sam2.1-hiera-small` | IoU |

All metrics are **higher-is-better**. Promotion happens automatically when a
run beats the current master.

## Setup

```bash
uv sync
huggingface-cli login
```

## Quick Start

### Object Detection (CPPE-5)

```bash
uv run scripts/refresh_master.py
# edit configs/base_detect.yaml (one knob change)
uv run prepare.py --dataset cppe-5 --task detect --split train
uv run scripts/hf_job.py preflight --task detect
uv run scripts/hf_job.py launch --task detect --config configs/base_detect.yaml
uv run scripts/hf_job.py logs <JOB_ID> --follow --output /tmp/vision-run.log
uv run scripts/parse_metric.py /tmp/vision-run.log
uv run scripts/submit_patch.py --comment "detect: <hypothesis>"
```

### Image Classification (Food-101)

```bash
uv run scripts/refresh_master.py
# edit configs/base_classify.yaml (one knob change)
uv run prepare.py --dataset food101 --task classify --split train
uv run scripts/hf_job.py launch --task classify --config configs/base_classify.yaml
uv run scripts/hf_job.py logs <JOB_ID> --follow --output /tmp/vision-run.log
uv run scripts/parse_metric.py /tmp/vision-run.log
uv run scripts/submit_patch.py --comment "classify: <hypothesis>"
```

### Segmentation

```bash
uv run scripts/refresh_master.py
# edit configs/base_segment.yaml (one knob change)
uv run prepare.py --dataset <name> --task segment --split train
uv run scripts/hf_job.py launch --task segment --config configs/base_segment.yaml
uv run scripts/hf_job.py logs <JOB_ID> --follow --output /tmp/vision-run.log
uv run scripts/parse_metric.py /tmp/vision-run.log
uv run scripts/submit_patch.py --comment "segment: <hypothesis>"
```

### Local Execution

```bash
uv run scripts/run_local.py --task detect --config configs/base_detect.yaml
```

## Operating Model

1. **Config YAML is the experiment surface.** Edit `configs/*.yaml`, never
   training scripts.
2. **One hypothesis per run.** Change exactly one config knob per experiment.
3. **Refresh before each experiment.** `uv run scripts/refresh_master.py`
   restores the config to the current promoted master.
4. **Record every run.** `uv run scripts/submit_patch.py --comment "..."`
   appends to `research/results.tsv` and promotes if the metric beats master.
5. **Local promoted master is the source of truth.** See
   `research/live/master.json` and `research/results.tsv`.

## Config Knobs

Experiments typically modify these YAML keys:

- `learning_rate`, `weight_decay`, `warmup_steps`, `lr_scheduler_type`
- `per_device_train_batch_size`, `gradient_accumulation_steps`
- `num_train_epochs`
- `image_size`, `use_albumentations`, `use_trivial_augment`
- `freeze_backbone`, `prompt_type`, `loss_type`

## Agent-Driven Sessions

Point an AI agent at `AGENTS.md` and `program.md` for the full operating
contract. Example prompts:

> Run a detection experiment on CPPE-5 testing lr=1e-4 vs the current master.

> Plan a classification campaign on Food-101 exploring augmentation strategies.

> What has been tried so far? Summarize research/results.tsv.

## Repo Layout

```
configs/              Base and experiment config YAMLs
train_detect.py       Object detection training (stable, do not edit)
train_classify.py     Image classification training (stable, do not edit)
train_segment.py      Segmentation training (stable, do not edit)
prepare.py            Dataset validation (never edit)
scripts/
  hf_job.py           HF Jobs launcher (preflight, launch, logs)
  submit_patch.py     Record run and promote
  refresh_master.py   Restore config from promoted master
  parse_metric.py     Parse metrics from training logs
  run_local.py        Local GPU execution
  local_results.py    Results ledger management
  worker_common.py    Worktree isolation
  trackio_reporter.py Experiment monitoring
  dataset_inspector.py Dataset format validation
  estimate_cost.py    Cost estimation
research/
  results.tsv         Append-only run ledger
  notes.md            Experiment notebook
  do-not-repeat.md    Failed experiment guidance
  paper-ideas.md      Literature-derived hypotheses
  live/               Current promoted master and DAG
  campaigns/          Active campaign docs
  experiments/        Experiment docs
  templates/          Campaign, experiment, do-not-repeat templates
```
