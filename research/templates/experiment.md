# Experiment: <short title>

## Campaign

- Campaign: `<theme>`
- Task type: `detect|classify|segment|detect_yolo|track_yolo|segment_yolo|classify_yolo|pose_yolo|obb_yolo`
- Backend: `transformers|ultralytics`
- Model: `<model_name>`
- Dataset: `<dataset_name>`

## Hypothesis

<One sentence. What single config change do you expect to help, and why?>

## Parent Context

- Parent config hash: `<hash>`
- Master metric at dispatch: `<metric_name> = <value>`
- Worker id: `<worker-id>`
- Worktree: `<worktree-path>`

## Single Variable

<What exact config knob is being tested? e.g. learning_rate: 1e-4 -> 5e-5>

## Expected Upside

<Why this might improve the promotion metric>

## Duplicate Check

<Why this is not a duplicate of an open or recent experiment>

## Runtime

- Log path: `<log-path>`
- Execution: `local|hf_jobs`
- Launcher: `uv run scripts/hf_job.py launch --task <task> --config <path>`

## Allowed Edit Scope

- Config YAML only (not training scripts)

## Run Plan

- Refresh local master with `uv run scripts/refresh_master.py`
- Modify config YAML with single change
- Run `uv run scripts/hf_job.py preflight --task <task>`
- Run `uv run scripts/hf_job.py launch --task <task> --config <path>`
- Stream logs
- Parse `uv run scripts/parse_metric.py <log-path>` (optional: `--task` / `--config` for contract validation)
- Record with `uv run scripts/submit_patch.py --comment "..."`

## Result

- Metric: `<metric_name> = <value>`
- Recorded locally: `yes|no`
- Promoted locally: `yes|no`
- Interpretation: `<one or two sentences>`
- Failure mode, if any: `<brief note>`

## Memory-Keeper Handoff

- One short note for `research/notes.md`: `<summary>`
- Any do-not-repeat update: `<summary or none>`
