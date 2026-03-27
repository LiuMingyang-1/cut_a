# Common Data Contracts

This document defines the method-agnostic semantics shared by `cut-a-lab`.

The goal is to keep the core stable while allowing each method to define its
own concrete input structure.

## Minimum Shared Semantics

Any method that produces span-level features must be able to map its records to
the following common fields:

- `sample_id: str`
- `span_id: str`
- `sample_label: int`
- `silver_label: int | null`

Optional but commonly useful fields:

- `silver_confidence: float | null`
- `route: str | null`
- `span_type: str | null`

## Required Invariants

### Sample Identity

- `sample_id` must be stable within a file.
- `sample_id` must uniquely identify the parent sample of every span row.

### Span Identity

- `span_id` must be stable within a file.
- `span_id` must uniquely identify one span row.
- If multiple methods are combined in a recipe, all methods must align on the
  same set of `span_id` values.

### Labels

- `sample_label` is binary and must be `0` or `1`.
- `silver_label` may be `0`, `1`, or `null`.
- Rows with `silver_label = null` are retained for prediction output but are
  excluded from span-level training loss.

### Feature Shapes

- Every method must output a fixed-width feature vector for every row in a
  single input file.
- Width may differ between methods.
- Width must not vary across rows within one file.

### Alignment Across Methods

If a recipe combines multiple methods:

- all methods must cover the exact same `span_id` set
- `sample_id` must match for the same `span_id`
- `sample_label` must match for the same `span_id`
- `silver_label` must match for the same `span_id`

The core runtime treats any mismatch as a contract error.

## Output Semantics

Training outputs are standardized regardless of method:

- `*.oof_predictions.jsonl` store span-level predictions with row metadata
- `*.metrics.json` store aggregated span/sample metrics
- `training_summary.json` stores recipe-wide comparison results
- `error_analysis.json` stores the selected model error breakdown

## Philosophy

The core defines only the minimum shared semantics needed to train and evaluate
span classifiers. Everything else belongs to a method contract.
