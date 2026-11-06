# eval/

- `golden.jsonl`: drafted + human-reviewed questions. Only `status: approved` items are used.
- `golden_ci.jsonl`: small stratified subset used by the CI gate (`make ci-subset`).
- `baseline.json`: CI-subset metrics the gate compares against (`make baseline`). Created only from a real run.
- `calibration/human_labels.jsonl`: your labels for judge calibration.
- `runs/`: per-run outputs (git-ignored; also logged to MLflow).

Item fields: `question`, `expected_answer`, `supporting_chunk_ids`, `category` (easy | multi_section | unanswerable).
