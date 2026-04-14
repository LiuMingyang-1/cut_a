# ParaRel Experiment SSH Report

## Scope

This repo is now prepared for two cloud-side experiment runs:

1. `scripts/experiments/self_consistency_sign.py`
   - Reuses existing ParaRel greedy cache and `combined_spans.jsonl`
   - Adds only sampled text generations for self-consistency pseudo labels
   - Writes:
     - `outputs/experiments/<model>/self_consistency/samples_{split}.jsonl`
     - `outputs/experiments/<model>/self_consistency/agreement_{split}.jsonl`
     - `outputs/experiments/<model>/self_consistency/summary.txt`
     - `outputs/experiments/<model>/self_consistency/pseudo_sign_vs_true_sign.png`

2. `scripts/experiments/run_third_model_layer_spectrum.py`
   - Runs one new-model ParaRel cache build
   - Reuses the existing three analysis scripts unchanged:
     - `validate_sign_flip.py`
     - `feature_correctness_spectrum.py`
     - `neighbor_layer_propagation.py`
   - Writes:
     - `outputs/experiments/<model-key>/summary.txt`
     - plus the normal outputs from the three existing analysis scripts

## Local Constraints

- Do not run heavyweight inference locally.
- Do not download model weights locally.
- Use cloud-side model paths that already exist on the remote machine.

## Validation Done Locally

No model inference was executed locally.

Validated with the repo `uv` environment:

- `uv run python -m unittest tests.test_self_consistency tests.test_layer_spectrum_summary`
- `uv run python -m unittest tests.test_r_tuning_inference tests.test_r_tuning_method_build`
- `uv run python scripts/experiments/self_consistency_sign.py --help`
- `uv run python scripts/experiments/run_third_model_layer_spectrum.py --help`

## Cloud Run Commands

### Task 1: Llama Self-Consistency

```bash
uv run python scripts/experiments/self_consistency_sign.py \
  --model-output-root outputs/experiments/llama-3.1-8b-instruct \
  --model-name-or-path /root/autodl-tmp/hf/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --k 10 \
  --temperature 0.7 \
  --top-p 0.95 \
  --prompt-batch-size 4
```

Notes:

- This script reads greedy cache from:
  - `outputs/experiments/llama-3.1-8b-instruct/inference/pararel/{train,id_test,ood_test}/samples.jsonl`
- This script reads cached features from:
  - `outputs/experiments/llama-3.1-8b-instruct/method_inputs/pararel/{train,id_test,ood_test}/combined_spans.jsonl`
- It does not extract hidden states again.
- If sampled outputs already exist and you only want to rebuild summary/agreements:

```bash
uv run python scripts/experiments/self_consistency_sign.py \
  --model-output-root outputs/experiments/llama-3.1-8b-instruct \
  --model-name-or-path /root/autodl-tmp/hf/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --reuse-existing-samples
```

Stop after the first full Llama table is produced.

### Task 2: Third Model Layer Spectrum

Preferred Mistral command:

```bash
uv run python scripts/experiments/run_third_model_layer_spectrum.py \
  --model-key mistral-7b-instruct \
  --model-name-or-path /root/autodl-tmp/hf/models/mistralai/Mistral-7B-Instruct-v0.3 \
  --allow-local-7b \
  --batch-size 4 \
  --max-new-tokens 16
```

If rerunning summary only after prior outputs already exist:

```bash
uv run python scripts/experiments/run_third_model_layer_spectrum.py \
  --model-key mistral-7b-instruct \
  --model-name-or-path /root/autodl-tmp/hf/models/mistralai/Mistral-7B-Instruct-v0.3 \
  --allow-local-7b \
  --skip-existing
```

This wrapper will:

1. Build ParaRel inference cache under `outputs/experiments/mistral-7b-instruct/inference`
2. Build method inputs under `outputs/experiments/mistral-7b-instruct/method_inputs`
3. Run the three existing analysis scripts
4. Write `outputs/experiments/mistral-7b-instruct/summary.txt`

Stop after the three yes/no answers and summary are produced.

## Expected Output Checks

### Task 1

Inspect:

- `outputs/experiments/llama-3.1-8b-instruct/self_consistency/summary.txt`
- `outputs/experiments/llama-3.1-8b-instruct/self_consistency/pseudo_sign_vs_true_sign.png`

Main rows in the table:

- `train-label sign (oracle)`
- `train-sign transfer`
- `propagation (minimal)`
- `oracle best single layer`
- `pseudo-sign (train infer)`
- `pseudo-sign (on-split infer)`

### Task 2

Inspect:

- `outputs/experiments/mistral-7b-instruct/summary.txt`
- `outputs/experiments/mistral-7b-instruct/sign_flip_validation/layer_correlation_matrices.png`
- `outputs/experiments/mistral-7b-instruct/feature_correctness_spectrum/feature_correctness_spectrum.png`
- `outputs/experiments/mistral-7b-instruct/neighbor_layer_propagation/propagation_signs.png`

The wrapper summary answers:

1. whether early/mid vs late Pearson is significantly negative
2. whether strongest spectrum layer shifts and flips sign on OOD
3. whether final-layer dominance holds on all splits
