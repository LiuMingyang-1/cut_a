# Method Contract: icr

The `icr` method is the first method migrated into `cut-a-lab`.

Version `0.1.0` accepts **prepared span-level vectors**, not raw sample-level
ICR tensors. This keeps the new project decoupled from upstream data
preparation pipelines.

## Accepted Input Shape

Input format: `jsonl`

One line represents one span row.

## Required Fields

- `sample_id: str`
- `span_id: str`
- `sample_label: int`
- one vector field:
  - preferred: `icr_vector: list[float]`
  - legacy alias: `span_vector: list[float]`

## Optional Fields

- `silver_label: int | null`
- `silver_confidence: float | null`
- `route: str | null`
- `span_type: str | null`
- `candidate_index: int | null`
- `source_sample_index: int | null`
- `window_size: int | null`

Additional fields are preserved as row metadata in prediction outputs.

## Vector Rules

- vectors must be 1D numeric arrays
- vector width must be identical across rows in the same file
- width does not have to be hard-coded by the core

## Alignment Rules

The `icr` method uses `span_id` as the row key and `sample_id` as the sample
group key.

If `icr` is combined with other methods in a future recipe, rows are aligned by
`span_id`.

## Example

```json
{
  "sample_id": "12:0",
  "span_id": "12:0:window:3",
  "sample_label": 1,
  "silver_label": 1,
  "silver_confidence": 0.94,
  "route": "tokenizer_windows",
  "span_type": "window",
  "icr_vector": [0.1, 0.2, 0.25, 0.4]
}
```

Legacy-compatible example:

```json
{
  "sample_id": "12:0",
  "span_id": "12:0:window:3",
  "sample_label": 1,
  "silver_label": 1,
  "span_vector": [0.1, 0.2, 0.25, 0.4]
}
```

## Notes

- Raw `icr_scores` matrices are intentionally out of scope for `0.1.0`.
- If you want to ingest raw sample-level ICR tensors later, add a dedicated
  preparation step or a second ICR method variant instead of coupling it into
  the project core.
- `icr` can be run alone or combined with other standalone methods such as
  `entropy` and `delta_entropy` in a recipe.
