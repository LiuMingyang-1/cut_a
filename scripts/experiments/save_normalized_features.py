"""Extract and save normalized feature matrices from ParaRel method inputs.

Reads combined_spans.jsonl for train / id_test / ood_test splits,
recomputes labels as contains-match (robust for open-ended generation),
and saves a single .npz with raw ICR + Entropy vectors and labels.

The output .npz is the input to layer_ablation_analysis.py and weight_analysis.py.

Usage (server or local after downloading method_inputs):

    uv run python scripts/experiments/save_normalized_features.py \\
        --method-input-root outputs/experiments/llama-3.1-8b-instruct/method_inputs/pararel \\
        --output-path outputs/experiments/llama-3.1-8b-instruct/normalized_features.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SPLITS = ("train", "id_test", "ood_test")
SPLIT_KEYS = ("tr", "id", "od")


def contains_match(generated: str, expected: str) -> bool:
    return str(expected).strip().lower() in str(generated).strip().lower()


def load_split(method_input_dir: Path, split_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (icr_matrix, entropy_matrix, labels) for one split."""
    path = method_input_dir / split_name / "combined_spans.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")

    icr_rows, ent_rows, labels = [], [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            # Recompute label as contains-match (handles ParaRel open-ended generation)
            label = int(not contains_match(r["generated_text"], r["expected_answer"]))
            icr_rows.append(r["span_vector"])      # ICR: cosine distances between adjacent layers
            ent_rows.append(r["entropy_vector"])   # Entropy: logit-lens per-layer entropy
            labels.append(label)

    icr = np.array(icr_rows, dtype=np.float32)
    ent = np.array(ent_rows, dtype=np.float32)
    lbl = np.array(labels, dtype=np.int32)
    return icr, ent, lbl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method-input-root",
        type=Path,
        required=True,
        help="Path to method_inputs/<dataset>/ directory, e.g. outputs/.../method_inputs/pararel",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Where to save the .npz file.",
    )
    args = parser.parse_args()

    arrays: dict[str, np.ndarray] = {}
    for split_name, key in zip(SPLITS, SPLIT_KEYS):
        print(f"Loading {split_name}...", end=" ")
        icr, ent, lbl = load_split(args.method_input_root, split_name)
        arrays[f"{key}_icr_raw"] = icr
        arrays[f"{key}_ent_raw"] = ent
        arrays[f"{key}_lbl"] = lbl
        pos_rate = lbl.mean()
        print(f"{len(lbl)} samples  ICR={icr.shape}  Ent={ent.shape}  hallucinated={pos_rate:.3f}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_path, **arrays)
    print(f"\nSaved {len(arrays)} arrays to {args.output_path}")
    print("Keys:", list(arrays.keys()))


if __name__ == "__main__":
    main()
