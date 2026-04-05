# R-Tuning Multi-Dataset Pipeline Plan

Last updated: 2026-04-04

## Goal

Build a reusable multi-dataset pipeline with three clean stages:

1. Inference/cache stage
2. Method-build stage
3. MLP-only evaluation stage

The key requirement is to avoid rerunning large-model inference when doing
later hallucination analysis.

## User Requirements

- Use more datasets than HaluEval.
- The datasets are already copied under `data/R-Tuning-data/`.
- Run these feature-set ablations:
  - `icr_only`
  - `entropy_only`
  - `delta_entropy_only`
  - `icr_entropy`
  - `icr_delta_entropy`
- Remove downstream logistic-regression / sklearn baselines.
- Use a simple MLP only.
- Prefer GPU execution.
- Save enough layer-level useful data so later analysis does not require
  rerunning inference.
- Keep code placement aligned with the current project structure.
- Keep inference/state caching separated from downstream hallucination-method
  analysis.

## Current Repo Reality

The current repo already supports:

- method loaders for prepared span-level vectors
- feature-set recipes
- MLP and sklearn training
- artifact writing for training / analysis

The current repo does not yet support:

- reading raw `R-Tuning-data/*` datasets directly
- running one unified model inference pass and caching reusable model states
- deriving `icr`, `entropy`, and `delta_entropy` from cached inference outputs
- batch execution across datasets

## Dataset Inventory

Observed local files under `data/R-Tuning-data/`:

- `HaluEvalQA/HaluEvalQA.json` with 10000 rows
- `HotpotQA/hotpot_10k.json` with 10000 rows
- `HotpotQA/hotpot_test.json` with 7405 rows
- `FEVER/fever_10k.json` with 9999 rows
- `FEVER/fever_10k_test.json` with 9999 rows
- `WiCE/wice_train.json` with 4419 rows
- `WiCE/wice_test.json` with 958 rows
- `MMLU/MMLU_ID_train.json` with 28 rows
- `MMLU/MMLU_ID_test.json` with 28 rows
- `MMLU/MMLU_OOD_test.json` with 29 rows
- `pararel/training_data.json` with 5575 rows
- `pararel/ID_test_pararel.json` with 5584 rows
- `pararel/OOD_test_pararel.json` with 13974 rows

Observed example schemas:

- `HaluEvalQA`: JSONL records with `knowledge`, `question`,
  `right_answer`, `hallucinated_answer`
- `HotpotQA`: QA-style JSON array records with `question`, `answer`,
  `context`, `supporting_facts`
- `FEVER`: claim-verification style JSON array records with `claim`,
  `evidence`, `label`
- `WiCE`: claim-verification style JSON array records with `claim`,
  `evidence`, `label`, `supporting_sentences`
- `MMLU`: list-style MCQ rows
- `pararel`: list-style factual prompt rows

## Proposed Architecture

### Stage 1: Inference Cache

Add a new package:

- `src/cut_a_lab/prep/r_tuning/`

Planned responsibilities:

- dataset adapters:
  - map each raw dataset into a common `sample` representation
- prompt assembly:
  - define the model input text needed for one forward pass
- inference runner:
  - run a language model on GPU
- cache writer:
  - persist reusable state for later method construction

Planned output root:

- `outputs/r_tuning/inference/<dataset>/<split>/`

Planned cached artifacts per dataset/split:

- `manifest.json`
- `samples.jsonl`
- `spans.jsonl`
- `layer_cache.npz`
- `logits_cache.npz`
- `token_alignment.jsonl`

`layer_cache.npz` should contain reusable span- or token-aggregated layer
representations instead of forcing future analysis to go back to the base
model. Full raw token hidden states are intentionally not the default because
they will explode storage size. If needed later, add a separate opt-in flag.

### Stage 2: Method Build

Build method inputs from inference cache:

- `icr`
- `entropy`
- `delta_entropy`

Planned output root:

- `outputs/r_tuning/method_inputs/<dataset>/<split>/`

Planned artifacts:

- `icr_spans.jsonl`
- `entropy_spans.jsonl`
- `delta_entropy_spans.jsonl`
- `combined_spans.jsonl`
- `method_manifest.json`

### Stage 3: Evaluation

Reuse the existing training runtime where possible, but narrow it to:

- recipe feature sets:
  - `icr_only`
  - `entropy_only`
  - `delta_entropy_only`
  - `icr_entropy`
  - `icr_delta_entropy`
- model family:
  - torch simple MLP only

Planned output root:

- `outputs/r_tuning/training/<dataset>/<split>/`

## Code Placement Plan

- `src/cut_a_lab/prep/r_tuning/`
  - new raw-data adaptation and inference-cache code
- `src/cut_a_lab/core/`
  - only shared runtime/artifact extensions
- `src/cut_a_lab/recipes/`
  - recipe cleanup to the five requested feature sets
- `scripts/`
  - one runner for inference/cache
  - one runner for cache-to-method-input
  - one runner for method-input-to-MLP-eval
- `tests/`
  - adapter, cache, method-build, and multi-dataset summary coverage

## Progress Log

### 2026-04-04

- Confirmed the current repo only consumes prepared span-level vectors.
- Confirmed raw `R-Tuning-data` files are present locally and need a new
  preparation layer.
- Confirmed current training still runs both sklearn and torch; this needs to
  be narrowed for the requested workflow.
- Confirmed the current default recipe still contains `discrepancy_combined`;
  this has now been removed from the default evaluation path.
- Started writing the implementation plan into the repo to avoid context loss.
- Added `src/cut_a_lab/prep/r_tuning/` with:
  - dataset normalization
  - cache contracts
  - cache IO
  - inference helpers
  - cache-to-method-input conversion
- Added three stage-separated scripts:
  - `scripts/build_r_tuning_inference_cache.py`
  - `scripts/build_r_tuning_method_inputs.py`
  - `scripts/run_r_tuning_mlp_eval.py`
- Narrowed the default recipe to the five requested feature sets.
- Added `run_recipe(..., family_groups=...)` so batch evaluation can run
  torch-only MLP without sklearn baselines.
- Added feature-bundle artifact persistence per feature set:
  - `feature_bundle.npz`
  - `feature_bundle.json`
- Added tests for dataset discovery/normalization and cache-to-method-input
  conversion.
- Verified:
  - targeted unit tests pass
  - new modules compile
  - torch-only end-to-end smoke run works with the five requested feature sets
- Added batch-run quality-of-life improvements:
  - `--max-samples` on inference cache for smoke tests
  - `--skip-missing` on method-build and eval stages
  - cross-dataset best-model text summary generation
- Verified the batch scripts now degrade cleanly when upstream cache or method
  inputs are absent, instead of failing by default during smoke setup.
- Installed `pip` into the local `.venv` via `ensurepip`.
- Installed `transformers` and `sentencepiece` into the local `.venv`.
- Installed `socksio` because the local environment routes Hub traffic through
  `all_proxy=socks5://127.0.0.1:1280`.
- Verified import-time environment:
  - `torch 2.11.0`
  - `transformers 5.5.0`
  - `sentencepiece 0.2.1`
  - `mps` available
  - `cuda` unavailable
- Ran a real tiny-model smoke inference:
  - model: `sshleifer/tiny-gpt2`
  - dataset: `HaluEvalQA/default`
  - sample cap: `2`
  - result: inference cache written successfully
- Ran the downstream smoke stages on that cache:
  - method-input build succeeded
  - MLP eval completed without crashing
  - because both smoke examples had the same inferred label, training now
    records a clean `skipped` status instead of raising an exception

## Local Environment Notes

- Local `.venv` currently has:
  - `torch 2.11.0`
  - `numpy 2.2.6`
  - `scikit-learn 1.7.2`
- Local `.venv` currently has:
  - `transformers 5.5.0`
  - `sentencepiece 0.2.1`
- Local machine exposes:
  - `mps` available
  - `cuda` unavailable in this environment

This means the new inference-cache stage now has its Python dependencies
installed and has been exercised against a tiny Hugging Face smoke model in the
current environment.

## Runtime Guardrails

- Do not run a local 7B inference job by accident during smoke testing.
- The inference-cache script now refuses model names/paths that look like `7B`
  unless `--allow-local-7b` is passed explicitly.

## Current CLI Workflow

Stage 1, inference cache:

```bash
.venv/bin/python scripts/build_r_tuning_inference_cache.py \
  --model-name-or-path <small-model> \
  --dataset HaluEvalQA \
  --split default \
  --max-samples 32
```

Notes:

- use `--max-samples` for smoke tests
- do not use a local 7B checkpoint unless explicitly intended

Stage 2, cache to method inputs:

```bash
.venv/bin/python scripts/build_r_tuning_method_inputs.py \
  --dataset HaluEvalQA \
  --split default \
  --skip-missing
```

Stage 3, method inputs to MLP-only evaluation:

```bash
.venv/bin/python scripts/run_r_tuning_mlp_eval.py \
  --dataset HaluEvalQA \
  --split default \
  --skip-missing
```

The evaluation stage now writes both:

- `training_summary_all_datasets.json`
- `training_summary_all_datasets.txt`

for easy cross-dataset scanning.

## Immediate Next Steps

1. Run the inference-cache stage on one small dataset split first.
2. Inspect one real cache and confirm the saved layer tensors are sufficient for
   planned hallucination analysis.
3. If needed, extend cache contents with additional per-layer statistics before
   large-scale runs.
4. Run a slightly larger smoke set where both labels are present, so MLP
   metrics are non-skipped.
5. Run multi-dataset method build and MLP evaluation end to end.
