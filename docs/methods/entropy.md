# Method Contract: entropy

The `entropy` method models prepared span-level entropy vectors as a standalone
feature method.

Version `0.1.0` accepts **prepared span-level vectors**, not raw token-level or
sample-level entropy tensors.

## Accepted Input Shape

Input format: `jsonl`

One line represents one span row.

## Required Fields

- `sample_id: str`
- `span_id: str`
- `sample_label: int`
- `entropy_vector: list[float]`

## Optional Fields

- `silver_label: int | null`
- `silver_confidence: float | null`
- `route: str | null`
- `span_type: str | null`
- `sample_entropy_vector: list[float] | null`
- `candidate_index: int | null`
- `source_sample_index: int | null`
- `window_size: int | null`

Additional fields are preserved as row metadata in prediction outputs.

## Vector Rules

- vectors must be 1D numeric arrays
- vector width must be identical across rows in the same file
- width does not have to be hard-coded by the core

## Alignment Rules

The `entropy` method uses `span_id` as the row key and `sample_id` as the
sample group key.

If `entropy` is combined with other methods in a recipe, rows are aligned by
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
  "entropy_vector": [2.1, 2.0, 1.8, 1.7, 1.6]
}
```

## Notes

- Entropy is treated as a first-class method, not a mandatory companion to ICR.
- If you later want raw entropy tensor ingestion, add a separate preparation
  stage or a second entropy method variant.
