# Method Contract: delta_entropy

The `delta_entropy` method models prepared span-level delta entropy vectors as a
standalone feature method.

Version `0.1.0` accepts **prepared span-level vectors**, not raw entropy tensors
that are differenced inside the core.

## Accepted Input Shape

Input format: `jsonl`

One line represents one span row.

## Required Fields

- `sample_id: str`
- `span_id: str`
- `sample_label: int`
- `delta_entropy_vector: list[float]`

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

The `delta_entropy` method uses `span_id` as the row key and `sample_id` as the
sample group key.

If `delta_entropy` is combined with other methods in a recipe, rows are aligned
by `span_id`.

## Example

```json
{
  "sample_id": "12:0",
  "span_id": "12:0:window:3",
  "sample_label": 1,
  "silver_label": 1,
  "delta_entropy_vector": [0.1, -0.2, -0.1, -0.05]
}
```

## Notes

- Delta entropy is treated as its own method contract.
- The core does not assume whether delta entropy was produced from entropy,
  logits, or some other upstream transform.
