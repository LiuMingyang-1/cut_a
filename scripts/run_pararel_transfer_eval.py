"""Train MLP on pararel/train, evaluate on pararel/id_test and pararel/ood_test.

Output root: outputs/pararel_experiment/training/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cut_a_lab.core.feature_views import build_feature_set_bundle
from cut_a_lab.core.io import dump_json
from cut_a_lab.core.registry import get_method, get_recipe
from cut_a_lab.core.transfer_eval import run_transfer_eval
from cut_a_lab.models.torch_models import build_torch_model_factories


TEST_SPLITS = ("id_test", "ood_test")


def _load_feature_blocks(recipe_method_names: tuple[str, ...], method_input_dir: Path) -> dict:
    blocks = {}
    for method_name in recipe_method_names:
        path = method_input_dir / f"{method_name}_spans.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Method input not found: {path}")
        method = get_method(method_name)
        blocks[method_name] = method.load_feature_block(path)
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method-input-root",
        type=Path,
        default=Path("outputs/pararel_experiment/method_inputs"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/pararel_experiment/training"),
    )
    parser.add_argument("--recipe", default="cut_a_default")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    recipe = get_recipe(args.recipe)
    model_factories = build_torch_model_factories()

    train_dir = args.method_input_root / "pararel" / "train"
    print(f"Loading train method inputs from: {train_dir}")
    train_blocks = _load_feature_blocks(recipe.method_names, train_dir)

    summary_rows: list[dict[str, Any]] = []

    for feature_set in recipe.feature_sets:
        train_records, train_features, feature_names = build_feature_set_bundle(
            feature_set,
            [train_blocks[m] for m in feature_set.methods],
        )

        for test_split in TEST_SPLITS:
            test_dir = args.method_input_root / "pararel" / test_split
            if not test_dir.exists():
                print(f"[skip] missing method inputs for {test_split}: {test_dir}")
                continue

            try:
                test_blocks = _load_feature_blocks(recipe.method_names, test_dir)
            except FileNotFoundError as exc:
                print(f"[skip] {exc}")
                continue

            test_records, test_features, _ = build_feature_set_bundle(
                feature_set,
                [test_blocks[m] for m in feature_set.methods],
            )

            output_dir = args.output_root / feature_set.name / test_split
            results = run_transfer_eval(
                feature_set_name=feature_set.name,
                method_names=feature_set.methods,
                train_records=train_records,
                train_features=train_features,
                test_records=test_records,
                test_features=test_features,
                feature_names=feature_names,
                model_factories=model_factories,
                output_dir=output_dir,
                test_split_name=test_split,
                device=args.device,
                seed=args.seed,
            )
            summary_rows.append({
                "feature_set": feature_set.name,
                "test_split": test_split,
                "output_dir": str(output_dir.resolve()),
                "results": results,
            })

    dump_json(args.output_root / "transfer_eval_summary.json", {"runs": summary_rows})
    print(f"\nSummary written to {args.output_root / 'transfer_eval_summary.json'}")


if __name__ == "__main__":
    main()
