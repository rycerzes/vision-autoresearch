# Campaign: <theme>

## Theme

<What branch of inquiry does this campaign represent?>

## Task

- Task type: `detect|classify|segment`
- Backend: `transformers|ultralytics`
- Model: `<model_name>`
- Dataset: `<dataset_name>`
- Promotion metric: task-specific standard name (`mAP`, `mAP_50`, `mask_map`, `accuracy`, `mIoU`, … — must appear in `promotion:` and match `scripts/vision_lab/task_registry.py` for this task)

## Parent Context

- Parent config hash: `<hash>`
- Master metric at dispatch: `<metric_name> = <value>`

## Inclusion Rules

- All experiments belong to the same research theme.
- Each experiment tests one hypothesis only (one config knob change).
- Every experiment must be fresh relative to the current local promoted master.

## Candidate Experiments

- `<experiment-id>`: <short rationale>
- `<experiment-id>`: <short rationale>
- `<experiment-id>`: <short rationale>

## Exit Criteria

Pause or close this campaign when:

- the theme is exhausted
- local master changes invalidate the queued work
- recent runs show the branch is not promising

## Notes

<Short campaign summary, constraints, or follow-up guidance>
