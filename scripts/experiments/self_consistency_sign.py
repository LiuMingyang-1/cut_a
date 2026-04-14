"""Self-consistency pseudo-sign inference for ParaRel ICR features.

This experiment reuses existing greedy caches and method-input features.
It only adds sampled text generations to create agreement-based pseudo labels.

Example:
    uv run python scripts/experiments/self_consistency_sign.py \
        --model-output-root outputs/experiments/llama-3.1-8b-instruct \
        --model-name-or-path /root/autodl-tmp/hf/models/LLM-Research/Meta-Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cut_a_lab.analysis.self_consistency import (
    align_sample_metric,
    majority_vote,
    sanitize_correlation_values,
)
from cut_a_lab.core.evaluation import roc_auc_binary
from cut_a_lab.core.io import load_json, read_jsonl, write_jsonl
from cut_a_lab.prep.r_tuning.inference import ModelRunnerConfig, _load_model_and_tokenizer
try:
    from pararel_icr_common import (
        contains_match_correct,
        effective_auroc,
        load_icr_split,
        pearson_vector,
        per_layer_error_auroc,
        signs_from_values,
        spearman_vector,
        zscore_with_reference,
    )
except ModuleNotFoundError:
    from scripts.experiments.pararel_icr_common import (
        contains_match_correct,
        effective_auroc,
        load_icr_split,
        pearson_vector,
        per_layer_error_auroc,
        signs_from_values,
        spearman_vector,
        zscore_with_reference,
    )


DEFAULT_SPLITS = ("train", "id_test", "ood_test")
DEFAULT_EVAL_SPLITS = ("id_test", "ood_test")


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-output-root",
        type=Path,
        default=Path("outputs/experiments/llama-3.1-8b-instruct"),
        help="Model experiment root containing inference/ and method_inputs/ subdirs.",
    )
    parser.add_argument("--inference-root", type=Path, default=None)
    parser.add_argument("--method-input-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--eval-splits", nargs="+", default=list(DEFAULT_EVAL_SPLITS))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--prompt-batch-size",
        type=int,
        default=4,
        help="Number of base prompts per batch before expanding by k.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--label-source",
        choices=("auto_sample", "contains_match"),
        default="auto_sample",
    )
    parser.add_argument(
        "--corr-method",
        choices=("spearman", "pearson"),
        default="spearman",
        help="Primary pseudo-sign correlation. Default stays on Spearman.",
    )
    parser.add_argument(
        "--reuse-existing-samples",
        action="store_true",
        help="Reuse existing samples_{split}.jsonl files instead of resampling.",
    )
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    inference_root = args.inference_root or (args.model_output_root / "inference" / "pararel")
    method_input_root = args.method_input_root or (args.model_output_root / "method_inputs" / "pararel")
    output_dir = args.output_dir or (args.model_output_root / "self_consistency")
    return inference_root, method_input_root, output_dir


def _normalize_split_config(args: argparse.Namespace) -> None:
    args.splits = _unique_preserve_order(list(args.splits))
    args.eval_splits = _unique_preserve_order(list(args.eval_splits))
    if args.train_split not in args.splits:
        raise ValueError(f"train_split={args.train_split!r} must be included in --splits.")
    missing_eval = [split_name for split_name in args.eval_splits if split_name not in args.splits]
    if missing_eval:
        raise ValueError(f"All --eval-splits must also appear in --splits. Missing: {missing_eval}")


def _load_cached_split_samples(inference_root: Path, split_name: str) -> list[dict[str, Any]]:
    path = inference_root / split_name / "samples.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing greedy cache rows: {path}")
    return read_jsonl(path)


def _resolve_model_name_or_path(args: argparse.Namespace, inference_root: Path, train_split: str) -> str:
    if args.model_name_or_path:
        return str(args.model_name_or_path)
    manifest = load_json(inference_root / train_split / "manifest.json")
    value = manifest.get("model_name_or_path")
    if not value:
        raise KeyError("Could not infer model_name_or_path from manifest.json.")
    return str(value)


def _resolve_max_new_tokens(args: argparse.Namespace, inference_root: Path, train_split: str) -> int:
    if args.max_new_tokens is not None:
        return int(args.max_new_tokens)
    manifest = load_json(inference_root / train_split / "manifest.json")
    value = manifest.get("max_new_tokens")
    if value is None:
        raise KeyError("Could not infer max_new_tokens from manifest.json.")
    return int(value)


def _trim_generated_text(tokenizer: Any, generated_ids: Any, *, padded_prompt_len: int) -> str:
    answer_ids = generated_ids[padded_prompt_len:]
    eos_positions = (answer_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
    answer_end = int(eos_positions[0]) + 1 if len(eos_positions) > 0 else int(answer_ids.shape[0])
    answer_ids = answer_ids[:answer_end]
    return tokenizer.decode(answer_ids, skip_special_tokens=True).strip()


def _sample_generations_for_split(
    *,
    rows: list[dict[str, Any]],
    output_path: Path,
    model: Any,
    tokenizer: Any,
    resolved_device: str,
    k: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    prompt_batch_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    import torch

    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}.")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tokenizer.padding_side = "left"
    sampled_rows: list[dict[str, Any]] = []

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    batches = [rows[i: i + prompt_batch_size] for i in range(0, len(rows), prompt_batch_size)]
    iterator = tqdm(batches, desc=f"self-consistency:{output_path.stem}", unit="batch", total=len(batches)) if tqdm else batches

    for batch in iterator:
        expanded_rows: list[dict[str, Any]] = []
        prompt_texts: list[str] = []
        for row in batch:
            for _ in range(k):
                expanded_rows.append(row)
                prompt_texts.append(str(row["prompt_text"]))

        encoded = tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=False)
        encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
        padded_prompt_len = int(encoded["input_ids"].shape[1])

        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        grouped: dict[str, list[dict[str, Any]]] = {}
        for expanded_index, row in enumerate(expanded_rows):
            sampled_text = _trim_generated_text(tokenizer, generated[expanded_index], padded_prompt_len=padded_prompt_len)
            grouped.setdefault(str(row["sample_id"]), []).append(
                {
                    "text": sampled_text,
                    "matches_gold": int(contains_match_correct(sampled_text, row["expected_answer"])),
                }
            )

        for row in batch:
            candidates = grouped[str(row["sample_id"])]
            sampled_rows.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "expected_answer": row["expected_answer"],
                    "canonical_generated_text": row["generated_text"],
                    "canonical_matches_gold": int(contains_match_correct(row["generated_text"], row["expected_answer"])),
                    "sampled_generations": candidates,
                }
            )

    write_jsonl(output_path, sampled_rows)
    return sampled_rows


def _build_agreement_rows(sample_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    agreement_rows: list[dict[str, Any]] = []
    for row in sample_rows:
        sampled_generations = row["sampled_generations"]
        vote = majority_vote([candidate["text"] for candidate in sampled_generations])
        majority_index = next(idx for idx, key in enumerate(vote.answer_keys) if key == vote.majority_key)
        majority_text = sampled_generations[majority_index]["text"]
        majority_matches_gold = int(sampled_generations[majority_index]["matches_gold"])
        sampled_match_rate = float(np.mean([candidate["matches_gold"] for candidate in sampled_generations]))
        agreement_rows.append(
            {
                "sample_id": row["sample_id"],
                "expected_answer": row["expected_answer"],
                "canonical_generated_text": row["canonical_generated_text"],
                "canonical_matches_gold": int(row["canonical_matches_gold"]),
                "majority_answer_key": vote.majority_key,
                "majority_answer_text": majority_text,
                "majority_count": vote.majority_count,
                "majority_tie_count": vote.tie_count,
                "agreement_rate": vote.agreement_rate,
                "sampled_match_rate": sampled_match_rate,
                "majority_matches_gold": majority_matches_gold,
                "k": len(sampled_generations),
            }
        )
    return agreement_rows


def _correlation_vector(matrix: np.ndarray, target: np.ndarray, *, method: str) -> tuple[np.ndarray, np.ndarray]:
    if method == "spearman":
        return spearman_vector(matrix, target)
    if method == "pearson":
        return pearson_vector(matrix, target)
    raise ValueError(f"Unsupported corr-method {method!r}.")


def _aggregate_error_score(matrix_z: np.ndarray, correctness_signs: np.ndarray, layer_weights: np.ndarray) -> np.ndarray:
    normalizer = float(layer_weights.sum())
    if normalizer == 0.0:
        raise ValueError("Layer weights sum to zero.")
    return -((matrix_z * correctness_signs[None, :] * layer_weights[None, :]).sum(axis=1) / normalizer)


def _load_propagation_baseline(model_output_root: Path) -> dict[str, Any] | None:
    path = model_output_root / "neighbor_layer_propagation" / "neighbor_layer_propagation.json"
    if not path.exists():
        return None
    return load_json(path)


def _sign_accuracy(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.asarray(left) == np.asarray(right)))


def _plot_sign_comparison(
    *,
    output_path: Path,
    title_prefix: str,
    split_results: dict[str, dict[str, Any]],
    ordered_split_names: list[str],
) -> None:
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for split_name in ordered_split_names:
        rows.append(np.asarray(split_results[split_name]["true_signs"], dtype=np.float64))
        rows.append(np.asarray(split_results[split_name]["pseudo_signs"], dtype=np.float64))
        labels.append(f"{split_name} true")
        labels.append(f"{split_name} pseudo")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    image = ax.imshow(np.vstack(rows), cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Layer")
    ax.set_title("True vs pseudo sign tables")
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)

    ax = axes[1]
    layers = np.arange(len(rows[0]))
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(ordered_split_names))))
    ax.axhline(0.0, color="#111111", linewidth=1.0, alpha=0.6)
    for color, split_name in zip(colors, ordered_split_names):
        ax.plot(layers, split_results[split_name]["true_rho"], color=color, linewidth=1.8, label=f"{split_name} true")
        ax.plot(
            layers,
            split_results[split_name]["pseudo_rho"],
            color=color,
            linewidth=1.4,
            linestyle="--",
            label=f"{split_name} pseudo",
        )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Correlation")
    ax.set_title("True vs pseudo correlation spectra")
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    fig.suptitle(f"{title_prefix}: self-consistency pseudo-sign", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _build_summary_text(
    *,
    config: argparse.Namespace,
    output_dir: Path,
    propagation_baseline: dict[str, Any] | None,
    train_pseudo_signs: np.ndarray,
    train_true_signs: np.ndarray,
    split_results: dict[str, dict[str, Any]],
) -> str:
    ordered_eval_splits = [split_name for split_name in config.eval_splits if split_name in split_results]
    ordered_summary_splits = [split_name for split_name in config.splits if split_name in split_results]

    lines = [
        "Self-consistency pseudo-sign summary",
        f"Output dir: {output_dir.resolve()}",
        f"Correlation: {config.corr_method}",
        f"k={config.k} temperature={config.temperature} top_p={config.top_p} max_new_tokens={config.max_new_tokens}",
        f"Label source: {config.label_source}",
        "",
        f"Train pseudo-vs-true sign accuracy: {_sign_accuracy(train_pseudo_signs, train_true_signs):.4f}",
        "",
    ]

    header = f"{'method':28s}"
    for split_name in ordered_eval_splits:
        header += f" {split_name:>12s}"
    lines.append(header)

    table_rows: list[tuple[str, list[float]]] = [
        (
            "train-label sign (oracle)",
            [float(split_results[split_name]["oracle_weighted_auc"]) for split_name in ordered_eval_splits],
        ),
        (
            "train-sign transfer",
            [float(split_results[split_name]["train_sign_transfer_auc"]) for split_name in ordered_eval_splits],
        ),
        (
            "propagation (minimal)",
            [
                float(propagation_baseline["target_splits"][split_name]["propagated_auc"])
                if propagation_baseline and split_name in propagation_baseline.get("target_splits", {})
                else float("nan")
                for split_name in ordered_eval_splits
            ],
        ),
        (
            "oracle best single layer",
            [float(split_results[split_name]["best_single_effective_auc"]) for split_name in ordered_eval_splits],
        ),
        (
            "pseudo-sign (train infer)",
            [float(split_results[split_name]["pseudo_train_transfer_auc"]) for split_name in ordered_eval_splits],
        ),
        (
            "pseudo-sign (on-split infer)",
            [float(split_results[split_name]["pseudo_on_split_auc"]) for split_name in ordered_eval_splits],
        ),
    ]
    for name, values in table_rows:
        row = f"{name:28s}"
        for value in values:
            row += f" {value:12.4f}"
        lines.append(row)

    lines.append("")
    for split_name in ordered_summary_splits:
        payload = split_results[split_name]
        lines.extend(
            [
                (
                    f"[{split_name}] n={payload['n_samples']} "
                    f"agreement_mean={payload['agreement_mean']:.4f} agreement_std={payload['agreement_std']:.4f}"
                ),
                (
                    f"  pseudo best |corr|: L{payload['best_pseudo_abs_layer']} "
                    f"({payload['pseudo_rho'][payload['best_pseudo_abs_layer']]:+.4f})"
                ),
                (
                    f"  true   best |corr|: L{payload['best_true_abs_layer']} "
                    f"({payload['true_rho'][payload['best_true_abs_layer']]:+.4f})"
                ),
                f"  pseudo-vs-true sign accuracy: {payload['pseudo_sign_accuracy_vs_true']:.4f}",
                "",
            ]
        )

    lines.extend(
        [
            "Artifacts",
        ]
    )
    for split_name in ordered_summary_splits:
        lines.append(f"  samples[{split_name}]: {output_dir / f'samples_{split_name}.jsonl'}")
        lines.append(f"  agree[{split_name}]:   {output_dir / f'agreement_{split_name}.jsonl'}")
    lines.extend(
        [
            f"  Figure:        {output_dir / 'pseudo_sign_vs_true_sign.png'}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    _normalize_split_config(args)
    inference_root, method_input_root, output_dir = _resolve_paths(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    title_prefix = args.title or output_dir.name

    model_name_or_path = _resolve_model_name_or_path(args, inference_root, args.train_split)
    max_new_tokens = _resolve_max_new_tokens(args, inference_root, args.train_split)
    args.max_new_tokens = max_new_tokens

    sample_rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split_name in args.splits:
        sample_path = output_dir / f"samples_{split_name}.jsonl"
        if args.reuse_existing_samples and sample_path.exists():
            sample_rows = read_jsonl(sample_path)
        else:
            sample_rows_by_split[split_name] = _load_cached_split_samples(inference_root, split_name)

    model = tokenizer = resolved_device = None
    if sample_rows_by_split:
        config = ModelRunnerConfig(
            model_name_or_path=model_name_or_path,
            device=args.device,
            max_new_tokens=max_new_tokens,
            batch_size=args.prompt_batch_size,
        )
        model, tokenizer, resolved_device = _load_model_and_tokenizer(config)

    agreement_rows_by_split: dict[str, list[dict[str, Any]]] = {}
    split_results: dict[str, dict[str, Any]] = {}

    for split_name in args.splits:
        sample_path = output_dir / f"samples_{split_name}.jsonl"
        agreement_path = output_dir / f"agreement_{split_name}.jsonl"

        if args.reuse_existing_samples and sample_path.exists():
            sample_rows = read_jsonl(sample_path)
        else:
            sample_rows = _sample_generations_for_split(
                rows=sample_rows_by_split[split_name],
                output_path=sample_path,
                model=model,
                tokenizer=tokenizer,
                resolved_device=resolved_device,
                k=args.k,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=max_new_tokens,
                prompt_batch_size=args.prompt_batch_size,
                seed=args.seed,
            )
        sample_rows_by_split[split_name] = sample_rows

        agreement_rows = _build_agreement_rows(sample_rows)
        write_jsonl(agreement_path, agreement_rows)
        agreement_rows_by_split[split_name] = agreement_rows

        split = load_icr_split(method_input_root, split_name, label_source=args.label_source)
        true_rho, _ = spearman_vector(split.matrix, split.correctness)
        true_rho = sanitize_correlation_values(true_rho)
        true_signs = signs_from_values(true_rho)
        true_weights = np.abs(true_rho)

        agreement_by_id = {str(row["sample_id"]): float(row["agreement_rate"]) for row in agreement_rows}
        agreement_vector = align_sample_metric(split.sample_ids, agreement_by_id, name="agreement_rate")
        pseudo_rho, _ = _correlation_vector(split.matrix, agreement_vector, method=args.corr_method)
        pseudo_rho = sanitize_correlation_values(pseudo_rho)
        pseudo_signs = signs_from_values(pseudo_rho)
        pseudo_weights = np.abs(pseudo_rho)

        raw_error_auc = per_layer_error_auroc(split.matrix, split.error)
        best_single_effective_auc = float(np.max(effective_auroc(raw_error_auc)))

        split_results[split_name] = {
            "n_samples": int(split.matrix.shape[0]),
            "sample_ids": split.sample_ids,
            "matrix": split.matrix,
            "error": split.error,
            "true_rho": true_rho.tolist(),
            "true_signs": true_signs.astype(int).tolist(),
            "true_weights": true_weights.tolist(),
            "pseudo_rho": pseudo_rho.tolist(),
            "pseudo_signs": pseudo_signs.astype(int).tolist(),
            "pseudo_weights": pseudo_weights.tolist(),
            "agreement_mean": float(np.mean(agreement_vector)),
            "agreement_std": float(np.std(agreement_vector)),
            "best_pseudo_abs_layer": int(np.argmax(np.abs(pseudo_rho))),
            "best_true_abs_layer": int(np.argmax(np.abs(true_rho))),
            "pseudo_sign_accuracy_vs_true": _sign_accuracy(pseudo_signs, true_signs),
            "oracle_weighted_auc": float("nan"),
            "train_sign_transfer_auc": float("nan"),
            "pseudo_train_transfer_auc": float("nan"),
            "pseudo_on_split_auc": float("nan"),
            "best_single_effective_auc": best_single_effective_auc,
        }

    train_payload = split_results[args.train_split]
    train_matrix = np.asarray(train_payload["matrix"], dtype=np.float64)
    train_true_signs = np.asarray(train_payload["true_signs"], dtype=np.float64)
    train_true_weights = np.asarray(train_payload["true_weights"], dtype=np.float64)
    train_pseudo_signs = np.asarray(train_payload["pseudo_signs"], dtype=np.float64)
    train_pseudo_weights = np.asarray(train_payload["pseudo_weights"], dtype=np.float64)

    for split_name in args.eval_splits:
        target_payload = split_results[split_name]
        target_matrix = np.asarray(target_payload["matrix"], dtype=np.float64)
        target_error = np.asarray(target_payload["error"], dtype=np.int32)
        target_true_signs = np.asarray(target_payload["true_signs"], dtype=np.float64)
        target_true_weights = np.asarray(target_payload["true_weights"], dtype=np.float64)
        target_pseudo_signs = np.asarray(target_payload["pseudo_signs"], dtype=np.float64)
        target_pseudo_weights = np.asarray(target_payload["pseudo_weights"], dtype=np.float64)

        target_z = zscore_with_reference(train_matrix, target_matrix)
        target_payload["oracle_weighted_auc"] = float(
            roc_auc_binary(target_error, _aggregate_error_score(target_z, target_true_signs, target_true_weights))
        )
        target_payload["train_sign_transfer_auc"] = float(
            roc_auc_binary(target_error, _aggregate_error_score(target_z, train_true_signs, train_true_weights))
        )
        target_payload["pseudo_train_transfer_auc"] = float(
            roc_auc_binary(target_error, _aggregate_error_score(target_z, train_pseudo_signs, train_pseudo_weights))
        )
        target_payload["pseudo_on_split_auc"] = float(
            roc_auc_binary(target_error, _aggregate_error_score(target_z, target_pseudo_signs, target_pseudo_weights))
        )

    propagation_baseline = _load_propagation_baseline(args.model_output_root)
    _plot_sign_comparison(
        output_path=output_dir / "pseudo_sign_vs_true_sign.png",
        title_prefix=title_prefix,
        split_results=split_results,
        ordered_split_names=args.splits,
    )
    summary_text = _build_summary_text(
        config=args,
        output_dir=output_dir,
        propagation_baseline=propagation_baseline,
        train_pseudo_signs=train_pseudo_signs,
        train_true_signs=train_true_signs,
        split_results=split_results,
    )
    (output_dir / "summary.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    main()
