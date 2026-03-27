"""CLI entrypoint for cut-a-lab."""

from __future__ import annotations

import argparse
from pathlib import Path

from cut_a_lab.analysis.error_analysis import run_error_analysis
from cut_a_lab.analysis.visualize import generate_figures
from cut_a_lab.core.artifacts import load_json_if_exists
from cut_a_lab.core.io import dump_json
from cut_a_lab.core.registry import get_method, get_recipe, list_methods, list_recipes
from cut_a_lab.core.training import run_recipe


def _parse_method_inputs(raw_items: list[str] | None) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for item in raw_items or []:
        method_name, separator, raw_path = item.partition("=")
        if not separator or not method_name or not raw_path:
            raise ValueError(
                f"Invalid --method-input value {item!r}. Expected the form <method>=<path>."
            )
        if method_name in parsed:
            raise ValueError(f"Duplicate method input provided for {method_name!r}.")
        parsed[method_name] = Path(raw_path).expanduser()
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available methods and recipes.")
    list_parser.set_defaults(func=_run_list)

    describe_method_parser = subparsers.add_parser("describe-method", help="Print one method contract.")
    describe_method_parser.add_argument("--method", required=True, help="Method name to describe.")
    describe_method_parser.set_defaults(func=_run_describe_method)

    describe_recipe_parser = subparsers.add_parser("describe-recipe", help="Print one recipe summary.")
    describe_recipe_parser.add_argument("--recipe", required=True, help="Recipe name to describe.")
    describe_recipe_parser.set_defaults(func=_run_describe_recipe)

    run_parser = subparsers.add_parser("run", help="Run one recipe.")
    run_parser.add_argument("--recipe", required=True, help="Recipe name to run.")
    run_parser.add_argument(
        "--method-input",
        action="append",
        default=[],
        help="Method input in the form <method>=<path>. Repeat once per method.",
    )
    run_parser.add_argument("--output-dir", type=Path, required=True, help="Directory for recipe outputs.")
    run_parser.add_argument("--baseline-dir", type=Path, default=None, help="Optional baseline predictions directory.")
    run_parser.add_argument("--device", type=str, default="cpu", help="Torch device for torch family training.")
    run_parser.add_argument("--skip-training", action="store_true", help="Reuse an existing training summary.")
    run_parser.add_argument("--skip-error-analysis", action="store_true", help="Skip error analysis.")
    run_parser.add_argument("--skip-figures", action="store_true", help="Skip figure generation.")
    run_parser.set_defaults(func=_run_recipe_command)

    return parser


def _run_list(_: argparse.Namespace) -> None:
    print("Methods:")
    for method_name in list_methods():
        print(f"- {method_name}")
    print("")
    print("Recipes:")
    for recipe_name in list_recipes():
        print(f"- {recipe_name}")


def _run_describe_method(args: argparse.Namespace) -> None:
    method = get_method(args.method)
    print(method.describe())


def _run_describe_recipe(args: argparse.Namespace) -> None:
    recipe = get_recipe(args.recipe)
    print(recipe.describe())


def _run_recipe_command(args: argparse.Namespace) -> None:
    recipe = get_recipe(args.recipe)
    method_inputs = _parse_method_inputs(args.method_input)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    training_dir = output_dir / "training"
    error_analysis_dir = output_dir / "error_analysis"
    figures_dir = output_dir / "figures"

    training_summary = None
    if args.skip_training:
        print("[1/4] Training skipped by request")
        training_summary = load_json_if_exists(training_dir / "training_summary.json")
    else:
        print("[1/4] Running recipe training")
        training_summary = run_recipe(
            recipe=recipe,
            method_inputs=method_inputs,
            output_dir=training_dir,
            device=args.device,
        )
        print(f"Saved training outputs to {training_dir}")

    error_analysis = None
    if args.skip_error_analysis:
        print("\n[2/4] Error analysis skipped by request")
        error_analysis = load_json_if_exists(error_analysis_dir / "error_analysis.json")
    elif not (training_dir / "training_summary.json").exists():
        print("\n[2/4] No training summary found; skipping error analysis")
    else:
        print("\n[2/4] Running error analysis")
        error_analysis = run_error_analysis(
            training_dir=training_dir,
            output_dir=error_analysis_dir,
            baseline_dir=None if args.baseline_dir is None else Path(args.baseline_dir).expanduser().resolve(),
        )
        print(f"Saved error analysis to {error_analysis_dir}")

    generated_figures = None
    if args.skip_figures:
        print("\n[3/4] Figure generation skipped by request")
    elif not (training_dir / "training_summary.json").exists():
        print("\n[3/4] No training summary found; skipping figures")
    elif not (error_analysis_dir / "error_analysis.json").exists():
        print("\n[3/4] No error analysis found; skipping figures")
    else:
        print("\n[3/4] Generating figures")
        generated_figures = generate_figures(
            run_summary_path=output_dir / "run_summary.json",
            training_summary_path=training_dir / "training_summary.json",
            error_analysis_path=error_analysis_dir / "error_analysis.json",
            output_dir=figures_dir,
        )
        print(f"Saved figures to {figures_dir}")

    if training_summary is None:
        training_summary = load_json_if_exists(training_dir / "training_summary.json")
    if error_analysis is None:
        error_analysis = load_json_if_exists(error_analysis_dir / "error_analysis.json")

    summary = {
        "recipe": recipe.name,
        "description": recipe.description,
        "method_inputs": {name: str(path) for name, path in method_inputs.items()},
        "training_dir": str(training_dir) if training_dir.exists() else None,
        "training_summary_path": str(training_dir / "training_summary.json")
        if (training_dir / "training_summary.json").exists()
        else None,
        "baseline_dir": None if args.baseline_dir is None else str(Path(args.baseline_dir).expanduser().resolve()),
        "error_analysis_dir": str(error_analysis_dir) if error_analysis_dir.exists() else None,
        "error_analysis_path": str(error_analysis_dir / "error_analysis.json")
        if (error_analysis_dir / "error_analysis.json").exists()
        else None,
        "figures_dir": str(figures_dir) if figures_dir.exists() else None,
        "figures": generated_figures,
        "best_model": training_summary.get("best_model") if training_summary is not None else None,
        "selected_models": error_analysis.get("selected_models") if error_analysis is not None else None,
    }
    dump_json(output_dir / "run_summary.json", summary)

    print("\n[4/4] Summary")
    print(f"Recipe: {recipe.name}")
    if summary["best_model"] is not None:
        best_model = summary["best_model"]
        print(
            "Best model: "
            f"{best_model['feature_set']} / {best_model['family_group']} / {best_model['model']} "
            f"(sample AUROC={best_model['sample_auroc']})"
        )
    if summary["training_summary_path"] is not None:
        print(f"Training summary: {summary['training_summary_path']}")
    if summary["error_analysis_path"] is not None:
        print(f"Error analysis: {summary['error_analysis_path']}")
    if summary["figures_dir"] is not None:
        print(f"Figures dir: {summary['figures_dir']}")
    print(f"Run summary: {output_dir / 'run_summary.json'}")


def main() -> None:
    """CLI entrypoint."""
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
