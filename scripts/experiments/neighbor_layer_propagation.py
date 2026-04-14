"""Anchor-assisted local neighbor-layer sign propagation for ParaRel ICR.

This is a minimal CPU-only calibration baseline:
  1. Use train split labels to estimate which layers are strong sign anchors.
  2. On each target split, build a local signed correlation graph from unlabeled
     ICR features.
  3. Propagate anchor signs through that local graph with iterative updates.
  4. Aggregate sign-calibrated layers into an error score and compare against:
     - direct train-sign transfer
     - target oracle sign assignment

Example:
    uv run python scripts/experiments/neighbor_layer_propagation.py \
        --method-input-root outputs/experiments/llama-3.1-8b-instruct/method_inputs/pararel \
        --output-dir outputs/experiments/llama-3.1-8b-instruct/neighbor_layer_propagation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from cut_a_lab.core.evaluation import roc_auc_binary
from pararel_icr_common import (
    load_icr_split,
    local_signed_correlation_graph,
    signs_from_values,
    spearman_vector,
    zscore_with_reference,
)


DEFAULT_TARGET_SPLITS = ("id_test", "ood_test")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-splits", nargs="+", default=list(DEFAULT_TARGET_SPLITS))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--anchor-top-k", type=int, default=8)
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument(
        "--label-source",
        choices=("auto_sample", "contains_match"),
        default="auto_sample",
    )
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def propagate_signs(
    anchor_signs: np.ndarray,
    anchor_weights: np.ndarray,
    graph_weights: np.ndarray,
    graph_signs: np.ndarray,
    *,
    max_iters: int,
) -> np.ndarray:
    state = np.where(anchor_weights > 0.0, anchor_signs, 1.0).astype(np.float64)
    for _ in range(max_iters):
        new_state = state.copy()
        for layer_idx in range(len(state)):
            field = (
                anchor_weights[layer_idx] * anchor_signs[layer_idx]
                + np.sum(graph_weights[layer_idx] * graph_signs[layer_idx] * state)
            )
            new_state[layer_idx] = 1.0 if field >= 0.0 else -1.0
        if np.array_equal(new_state, state):
            break
        state = new_state
    return state


def aggregate_error_score(matrix_z: np.ndarray, correctness_signs: np.ndarray, layer_weights: np.ndarray) -> np.ndarray:
    normalizer = float(layer_weights.sum())
    if normalizer == 0.0:
        raise ValueError("Layer weights sum to zero.")
    return -((matrix_z * correctness_signs[None, :] * layer_weights[None, :]).sum(axis=1) / normalizer)


def sign_matrix_rows(train_signs: np.ndarray, propagated_signs: np.ndarray, oracle_signs: np.ndarray) -> np.ndarray:
    return np.vstack([train_signs, propagated_signs, oracle_signs]).astype(np.float64)


def plot_signs_and_spectra(
    *,
    train_rho: np.ndarray,
    split_payloads: dict[str, dict[str, Any]],
    output_path: Path,
    title_prefix: str,
) -> None:
    fig, axes = plt.subplots(len(split_payloads), 2, figsize=(11, 4.0 * len(split_payloads)), squeeze=False)
    cmap = plt.get_cmap("coolwarm")

    for row_idx, (split_name, payload) in enumerate(split_payloads.items()):
        oracle_rho = np.asarray(payload["oracle_spearman_rho_correctness"], dtype=np.float64)
        layers = np.arange(len(train_rho))

        ax = axes[row_idx][0]
        ax.axhline(0.0, color="#111111", linewidth=1.0, alpha=0.6)
        ax.plot(layers, train_rho, "o-", color="#2563eb", linewidth=1.6, markersize=4, label="Train rho")
        ax.plot(layers, oracle_rho, "s-", color="#dc2626", linewidth=1.4, markersize=3.5, label=f"{split_name} rho")
        for anchor_layer in payload["anchor_layers"]:
            ax.axvline(anchor_layer + 0.5, color="#9ca3af", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.set_title(
            f"{split_name}: train vs target correctness rho\n"
            f"prop AUROC={payload['propagated_auc']:.3f}  oracle={payload['oracle_weighted_auc']:.3f}"
        )
        ax.set_xlabel("Layer")
        ax.set_ylabel("Spearman rho")
        ax.grid(alpha=0.2, linewidth=0.5)
        ax.legend(frameon=False, fontsize=8)

        ax = axes[row_idx][1]
        image = sign_matrix_rows(
            np.asarray(payload["train_correctness_signs"], dtype=np.float64),
            np.asarray(payload["propagated_correctness_signs"], dtype=np.float64),
            np.asarray(payload["oracle_correctness_signs"], dtype=np.float64),
        )
        ax.imshow(image, cmap=cmap, vmin=-1.0, vmax=1.0, aspect="auto")
        ax.set_yticks([0, 1, 2], labels=["train", "prop", "oracle"])
        ax.set_xlabel("Layer")
        ax.set_title(
            f"{split_name}: sign assignments\n"
            f"prop-vs-oracle={payload['propagated_sign_accuracy_vs_oracle']:.3f}"
        )

    fig.suptitle(f"{title_prefix}: neighbor-layer propagation", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_metric_bars(split_payloads: dict[str, dict[str, Any]], *, output_path: Path, title_prefix: str) -> None:
    split_names = list(split_payloads.keys())
    metric_names = [
        ("train_sign_transfer_auc", "Train-sign"),
        ("propagated_auc", "Propagated"),
        ("oracle_weighted_auc", "Oracle-weighted"),
        ("best_single_effective_auc", "Best single"),
    ]
    x = np.arange(len(split_names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for idx, (metric_key, label) in enumerate(metric_names):
        values = [split_payloads[split_name][metric_key] for split_name in split_names]
        ax.bar(x + (idx - 1.5) * width, values, width, label=label)

    ax.axhline(0.5, color="#111111", linewidth=1.0, alpha=0.6)
    ax.set_xticks(x, split_names)
    ax.set_ylabel("AUROC")
    ax.set_title(f"{title_prefix}: propagation AUROC comparison")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_summary_text(config: argparse.Namespace, train_payload: dict[str, Any], split_payloads: dict[str, dict[str, Any]]) -> str:
    lines = [
        "Neighbor-layer propagation summary",
        f"Method inputs: {config.method_input_root.resolve()}",
        f"Label source: {config.label_source}",
        f"Train split: {config.train_split}",
        f"Radius: {config.radius}",
        f"Anchor top-k: {config.anchor_top_k}",
        "",
        (
            f"[train] n={train_payload['n_samples']} layers={train_payload['n_layers']} "
            f"sample_label_mode={train_payload['sample_label_mode']} "
            f"sample-vs-contains-match={train_payload['sample_label_agreement_with_contains_match']:.3f}"
        ),
        f"  anchor layers: {train_payload['anchor_layers']}",
        f"  anchor train rho: {[round(v, 4) for v in train_payload['anchor_train_rho']]}",
        "",
    ]

    for split_name, payload in split_payloads.items():
        lines.extend(
            [
                (
                    f"[{split_name}] n={payload['n_samples']} "
                    f"sample_label_mode={payload['sample_label_mode']} "
                    f"sample-vs-contains-match={payload['sample_label_agreement_with_contains_match']:.3f}"
                ),
                f"  train-sign transfer AUROC: {payload['train_sign_transfer_auc']:.4f}",
                f"  propagated AUROC:          {payload['propagated_auc']:.4f}",
                f"  oracle-weighted AUROC:     {payload['oracle_weighted_auc']:.4f}",
                f"  best single effective:     {payload['best_single_effective_auc']:.4f}",
                f"  propagated sign acc:       {payload['propagated_sign_accuracy_vs_oracle']:.4f}",
                f"  train-sign acc:            {payload['train_sign_accuracy_vs_oracle']:.4f}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    title_prefix = args.title or args.output_dir.name

    train = load_icr_split(args.method_input_root, args.train_split, label_source=args.label_source)
    train_rho, _ = spearman_vector(train.matrix, train.correctness)
    train_signs = signs_from_values(train_rho)
    layer_weights = np.abs(train_rho)
    anchor_order = np.argsort(-layer_weights)
    anchor_layers = [int(layer) for layer in anchor_order[: args.anchor_top_k]]

    anchor_signs = np.zeros(train.matrix.shape[1], dtype=np.float64)
    anchor_weights = np.zeros(train.matrix.shape[1], dtype=np.float64)
    for layer_idx in anchor_layers:
        anchor_signs[layer_idx] = float(train_signs[layer_idx])
        anchor_weights[layer_idx] = float(layer_weights[layer_idx])

    train_payload = {
        "n_samples": int(train.matrix.shape[0]),
        "n_layers": int(train.matrix.shape[1]),
        "sample_label_mode": train.sample_label_mode,
        "sample_label_agreement_with_contains_match": train.sample_label_agreement_with_contains_match,
        "anchor_layers": anchor_layers,
        "anchor_train_rho": [float(train_rho[layer_idx]) for layer_idx in anchor_layers],
    }

    split_payloads: dict[str, dict[str, Any]] = {}
    for split_name in args.target_splits:
        target = load_icr_split(args.method_input_root, split_name, label_source=args.label_source)
        oracle_rho, _ = spearman_vector(target.matrix, target.correctness)
        oracle_signs = signs_from_values(oracle_rho)
        graph_weights, graph_signs = local_signed_correlation_graph(target.matrix, radius=args.radius, method="spearman")
        propagated_signs = propagate_signs(
            anchor_signs,
            anchor_weights,
            graph_weights,
            graph_signs,
            max_iters=args.max_iters,
        )

        target_z = zscore_with_reference(train.matrix, target.matrix)
        train_sign_score = aggregate_error_score(target_z, train_signs, layer_weights)
        propagated_score = aggregate_error_score(target_z, propagated_signs, layer_weights)
        oracle_score = aggregate_error_score(target_z, oracle_signs, layer_weights)
        best_single_effective_auc = float(
            np.max(
                np.maximum(
                    [roc_auc_binary(target.error, target.matrix[:, layer_idx]) for layer_idx in range(target.matrix.shape[1])],
                    [1.0 - roc_auc_binary(target.error, target.matrix[:, layer_idx]) for layer_idx in range(target.matrix.shape[1])],
                )
            )
        )

        payload = {
            "n_samples": int(target.matrix.shape[0]),
            "sample_label_mode": target.sample_label_mode,
            "sample_label_agreement_with_contains_match": target.sample_label_agreement_with_contains_match,
            "anchor_layers": anchor_layers,
            "train_correctness_signs": train_signs.astype(int).tolist(),
            "propagated_correctness_signs": propagated_signs.astype(int).tolist(),
            "oracle_correctness_signs": oracle_signs.astype(int).tolist(),
            "oracle_spearman_rho_correctness": oracle_rho.tolist(),
            "train_sign_transfer_auc": float(roc_auc_binary(target.error, train_sign_score)),
            "propagated_auc": float(roc_auc_binary(target.error, propagated_score)),
            "oracle_weighted_auc": float(roc_auc_binary(target.error, oracle_score)),
            "best_single_effective_auc": best_single_effective_auc,
            "train_sign_accuracy_vs_oracle": float(np.mean(train_signs == oracle_signs)),
            "propagated_sign_accuracy_vs_oracle": float(np.mean(propagated_signs == oracle_signs)),
        }
        split_payloads[split_name] = payload
        print(
            f"{split_name}: train-sign={payload['train_sign_transfer_auc']:.3f} "
            f"prop={payload['propagated_auc']:.3f} oracle={payload['oracle_weighted_auc']:.3f} "
            f"sign-acc={payload['propagated_sign_accuracy_vs_oracle']:.3f}"
        )

    plot_signs_and_spectra(
        train_rho=train_rho,
        split_payloads=split_payloads,
        output_path=args.output_dir / "propagation_signs",
        title_prefix=title_prefix,
    )
    plot_metric_bars(split_payloads, output_path=args.output_dir / "propagation_auroc_comparison", title_prefix=title_prefix)

    summary_payload = {
        "method_input_root": str(args.method_input_root.resolve()),
        "label_source": args.label_source,
        "train": train_payload,
        "target_splits": split_payloads,
        "config": {
            "radius": args.radius,
            "anchor_top_k": args.anchor_top_k,
            "max_iters": args.max_iters,
        },
    }
    summary_json_path = args.output_dir / "neighbor_layer_propagation.json"
    summary_txt_path = args.output_dir / "neighbor_layer_propagation_summary.txt"
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    summary_txt_path.write_text(build_summary_text(args, train_payload, split_payloads), encoding="utf-8")
    print(f"Saved summary json: {summary_json_path}")
    print(f"Saved summary txt:  {summary_txt_path}")


if __name__ == "__main__":
    main()
