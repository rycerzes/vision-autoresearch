# Vision Autoresearch

Multi-agent experiment loop for vision model finetuning. Adapted from [multiautoresearch](https://github.com/burtenshaw/multiautoresearch) - same disciplined single-change methodology, but for vision tasks with YAML configs as the experiment surface instead of editing training scripts directly.

Agents propose a hypothesis, change one config knob, run a finetune, and auto-promote when the metric beats the current master. Works locally on consumer GPUs or on HF Jobs.

## Tasks

| Task | Script | Default Model | Dataset | Headline metric |
|------|--------|---------------|---------|-----------------|
| Classify | `train_hf_vision.py` | `google/vit-base-patch16-224` | food101 | `accuracy` |
| Detect | `train_hf_vision.py` | `ustc-community/dfine-small-coco` | cppe-5 | `mAP` |
| Segment | `train_hf_vision.py` | `facebook/sam2.1-hiera-small` | — | `mIoU` |

Classification configs set `model_loader` and `adaptation_mode` (see `configs/base_classify.yaml`).

Ultralytics (`train_ultralytics.py`): `detect_yolo` / `track_yolo` / `pose_yolo` / `obb_yolo` default to `mAP` (with `mAP_50` allowed in `promotion:`); `segment_yolo` uses `mask_map`; `classify_yolo` uses `accuracy`.

Promotion metrics must be **standard names** allowed for that task (`scripts/vision_lab/metrics.py`, `scripts/vision_lab/task_registry.py`). Direction is **higher** or **lower** per metric, not universally “maximize.”

## Setup

```bash
uv sync
```

## Run locally (single GPU)

```bash
uv run scripts/refresh_master.py
# edit configs/base_classify.yaml — one knob change
uv run prepare.py --dataset food101 --task classify --split train
CUDA_VISIBLE_DEVICES=0 uv run scripts/run_local.py --task classify --config configs/base_classify.yaml
uv run scripts/submit_patch.py --comment "classify: lr 1e-4"
```

## Run on HF Jobs

```bash
uv run scripts/refresh_master.py
uv run scripts/hf_job.py launch --task classify --config configs/base_classify.yaml
uv run scripts/hf_job.py logs <JOB_ID> --follow --output /tmp/vision-run.log
uv run scripts/parse_metric.py /tmp/vision-run.log
uv run scripts/submit_patch.py --comment "classify: lr 1e-4"
```

## Agent-driven (opencode)

Send a single prompt — the agent handles refresh, config edit, run, parse, and submit:

```bash
CUDA_VISIBLE_DEVICES=0 opencode run "
Finetune google/vit-base-patch16-224 on food101 using the classify task.
Read AGENTS.md for repo conventions. Follow the standard workflow end-to-end.
"
```

For parallel experiments on multi-GPU:

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/opencode_worker.py run exp-01 &
CUDA_VISIBLE_DEVICES=1 uv run scripts/opencode_worker.py run exp-02 &
```

Each worker runs in an isolated git worktree under `.runtime/worktrees/`.

## How it works

1. **Config YAML is the experiment surface.** Edit `configs/*.yaml`, never training scripts. Configs are parsed natively via `HfArgumentParser.parse_yaml_file()` — keys map 1:1 to `TrainingArguments` / dataclass fields.
2. **One hypothesis per run.** Change exactly one config knob per experiment.
3. **Refresh before each experiment.** `refresh_master.py` restores configs to the current promoted master.
4. **Auto-promotion.** `submit_patch.py` appends to `research/results.tsv` and promotes the config if the metric beats master.
5. **`research/live/master.json`** is the source of truth.

## Config knobs

```yaml
learning_rate, weight_decay, warmup_steps, lr_scheduler_type
per_device_train_batch_size, gradient_accumulation_steps
num_train_epochs, fp16
model_loader, adaptation_mode (train_hf_vision tasks)
image_square_size (detect)
use_albumentations (detect), prompt_type (segment)
```

## Repo layout

```
.
├── configs/
│   ├── base_classify.yaml
│   ├── base_detect.yaml
│   └── base_segment.yaml
├── train_hf_vision.py         # stable — classify, detect, segment (do not edit)
├── prepare.py                 # dataset validation CLI (vision_lab.dataset_validation)
├── scripts/
│   ├── refresh_master.py      # restore config from promoted master
│   ├── run_local.py           # local GPU execution
│   ├── hf_job.py              # HF Jobs launcher
│   ├── parse_metric.py        # extract metrics from logs
│   ├── submit_patch.py        # record run + auto-promote
│   ├── opencode_worker.py     # agent worktree isolation
│   ├── worker_common.py       # shared worker utilities
│   ├── trackio_reporter.py    # experiment monitoring
│   ├── dataset_inspector.py   # dataset format validation
│   ├── estimate_cost.py       # cost estimation
│   └── local_results.py       # results ledger management
├── research/
│   ├── results.tsv            # append-only run ledger
│   ├── notes.md               # experiment notebook
│   ├── do-not-repeat.md       # failed experiment guidance
│   ├── paper-ideas.md         # literature-derived hypotheses
│   ├── live/                  # promoted master + DAG
│   ├── reference/             # seed master snapshots
│   ├── campaigns/             # active campaign docs
│   ├── experiments/           # per-experiment docs
│   └── templates/             # campaign/experiment templates
├── AGENTS.md                  # agent roles + rules
├── program.md                 # benchmark entrypoint
└── pyproject.toml
```

## Acknowledgments

Based on [multiautoresearch](https://github.com/burtenshaw/multiautoresearch) by [@burtenshaw](https://github.com/burtenshaw).
