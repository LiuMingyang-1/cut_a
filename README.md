# cut-a-lab

`cut-a-lab` is a clean standalone project for Cut A style experiments.

The project is intentionally method-oriented:

- `core` owns the experiment runtime, training, evaluation, and artifacts.
- `methods` own method-specific input contracts and feature construction.
- `recipes` define which methods are enabled for a concrete experiment.

Version `0.1.0` does **not** ingest raw upstream data pipelines. It consumes
already prepared method inputs. This keeps the project focused on experiment
logic instead of coupling it to a single upstream repository or scoring system.

## Design Principles

- No imports from `cuts/`, `shared/`, or `icr_probe_repro/`
- No `sys.path` mutation
- No fixed global input schema for every method
- Stable output artifacts
- Method contracts documented before implementation details

## Project Layout

```text
cut-a-lab/
├── README.md
├── pyproject.toml
├── configs/
│   └── examples/
├── docs/
│   ├── data-contracts.md
│   └── methods/
│       └── icr.md
├── src/
│   └── cut_a_lab/
│       ├── analysis/
│       ├── core/
│       ├── methods/
│       ├── models/
│       └── recipes/
└── tests/
```

## Core Vocabulary

`sample`
: The top-level prediction unit. Sample-level labels are used for grouped
cross-validation and sample-level aggregation.

`span`
: A sub-unit inside a sample. Training happens on span rows, while evaluation
is reported at both span level and sample level.

`method`
: A self-contained feature provider. Each method declares its own input
contract and builds its own feature block.

`recipe`
: A concrete experiment configuration that selects one or more methods and
defines feature-set combinations.

`feature block`
: The output of a method loader. It contains row-aligned records, a feature
matrix, and feature names.

## Common Data Semantics

The project does not require a universal input file format, but every method
must be able to produce row-aligned span records with the following minimum
semantics:

- `sample_id`: stable identifier for a sample
- `span_id`: stable identifier for a span
- `sample_label`: binary sample-level label
- `silver_label`: optional binary span-level label

See [docs/data-contracts.md](docs/data-contracts.md) for the method-agnostic
rules shared across methods.

## Method Contracts

Current methods:

- `icr`: [docs/methods/icr.md](docs/methods/icr.md)
- `entropy`: [docs/methods/entropy.md](docs/methods/entropy.md)
- `delta_entropy`: [docs/methods/delta_entropy.md](docs/methods/delta_entropy.md)

Each method document defines:

- accepted input file shape
- required and optional fields
- alignment rules
- vector constraints
- example records

## Current Recipe

The first recipe is `cut_a_default`.

For `0.1.0`, it runs the main prepared-vector ablations:

- `icr_only`
- `entropy_only`
- `delta_entropy_only`
- `icr_entropy`
- `icr_delta_entropy`
- `discrepancy_combined`

The point of this first recipe is to validate the new architecture while still
covering the main method combinations from the existing Cut A workflow.

## Outputs

Running the recipe writes:

```text
outputs/
├── training/
│   ├── training_summary.json
│   ├── comparison_table.txt
│   └── <feature_set>/<family>/*.json
├── error_analysis/
│   ├── error_analysis.json
│   └── error_analysis_summary.txt
├── figures/
└── run_summary.json
```

Expected stable artifacts:

- `training_summary.json`
- `*.metrics.json`
- `*.oof_predictions.jsonl`
- `error_analysis.json`

## CLI

List methods and recipes:

```bash
cut-a-lab list
```

Describe the `icr` method contract:

```bash
cut-a-lab describe-method --method icr
```

Run the default recipe with prepared method inputs:

```bash
cut-a-lab run \
  --recipe cut_a_default \
  --method-input icr=/path/to/icr_spans.jsonl \
  --method-input entropy=/path/to/entropy_spans.jsonl \
  --method-input delta_entropy=/path/to/delta_entropy_spans.jsonl \
  --output-dir /path/to/output \
  --device cpu
```

Compare against an external baseline prediction directory:

```bash
cut-a-lab run \
  --recipe cut_a_default \
  --method-input icr=/path/to/icr_spans.jsonl \
  --method-input entropy=/path/to/entropy_spans.jsonl \
  --method-input delta_entropy=/path/to/delta_entropy_spans.jsonl \
  --baseline-dir /path/to/baseline_predictions \
  --output-dir /path/to/output \
  --device cpu
```

## What This Project Does Not Do Yet

- raw upstream data extraction
- raw sample-to-span conversion
- raw signal derivation from sample-level tensors
- raw discrepancy feature derivation from sample-level tensors

Those can be added later as new methods or separate preparation stages without
changing the project core.
