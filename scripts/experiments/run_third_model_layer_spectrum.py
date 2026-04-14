"""Minimal orchestration for the third-model layer-spectrum experiment.

This wrapper runs the existing ParaRel cache build pipeline, then reuses the
existing three analysis scripts without changing their internal logic.

Example:
    uv run python scripts/experiments/run_third_model_layer_spectrum.py \
        --model-key mistral-7b-instruct \
        --model-name-or-path /root/autodl-tmp/hf/models/mistralai/Mistral-7B-Instruct-v0.3 \
        --allow-local-7b
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from cut_a_lab.analysis.layer_spectrum_summary import build_third_model_summary_text
from cut_a_lab.core.io import load_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/R-Tuning-data"))
    parser.add_argument("--model-key", default="mistral-7b-instruct")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--allow-local-7b", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def _run(cmd: list[str], *, workdir: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=workdir, check=True)


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_root = args.output_root or (Path("outputs/experiments") / args.model_key)
    output_root.mkdir(parents=True, exist_ok=True)

    inference_root = output_root / "inference"
    method_inputs_root = output_root / "method_inputs"
    pararel_method_input_root = method_inputs_root / "pararel"

    cache_summary = inference_root / "cache_summary.json"
    method_summary = method_inputs_root / "method_input_summary.json"
    sign_flip_json = output_root / "sign_flip_validation" / "sign_flip_validation.json"
    spectrum_json = output_root / "feature_correctness_spectrum" / "feature_correctness_spectrum.json"
    propagation_json = output_root / "neighbor_layer_propagation" / "neighbor_layer_propagation.json"

    if not (args.skip_existing and cache_summary.exists()):
        cmd = [
            sys.executable,
            "scripts/build_r_tuning_inference_cache.py",
            "--data-root",
            str(args.data_root),
            "--output-root",
            str(inference_root),
            "--model-name-or-path",
            str(args.model_name_or_path),
            "--device",
            str(args.device),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--batch-size",
            str(args.batch_size),
            "--dataset",
            "pararel",
            "--split",
            "train",
            "--split",
            "id_test",
            "--split",
            "ood_test",
        ]
        if args.allow_local_7b:
            cmd.append("--allow-local-7b")
        _run(cmd, workdir=repo_root)

    if not (args.skip_existing and method_summary.exists()):
        _run(
            [
                sys.executable,
                "scripts/build_r_tuning_method_inputs.py",
                "--data-root",
                str(args.data_root),
                "--cache-root",
                str(inference_root),
                "--output-root",
                str(method_inputs_root),
                "--dataset",
                "pararel",
                "--split",
                "train",
                "--split",
                "id_test",
                "--split",
                "ood_test",
            ],
            workdir=repo_root,
        )

    analyses = [
        (
            sign_flip_json,
            [
                sys.executable,
                "scripts/experiments/validate_sign_flip.py",
                "--method-input-root",
                str(pararel_method_input_root),
                "--output-dir",
                str(output_root / "sign_flip_validation"),
            ],
        ),
        (
            spectrum_json,
            [
                sys.executable,
                "scripts/experiments/feature_correctness_spectrum.py",
                "--method-input-root",
                str(pararel_method_input_root),
                "--output-dir",
                str(output_root / "feature_correctness_spectrum"),
            ],
        ),
        (
            propagation_json,
            [
                sys.executable,
                "scripts/experiments/neighbor_layer_propagation.py",
                "--method-input-root",
                str(pararel_method_input_root),
                "--output-dir",
                str(output_root / "neighbor_layer_propagation"),
            ],
        ),
    ]

    for artifact_path, cmd in analyses:
        if args.skip_existing and artifact_path.exists():
            continue
        _run(cmd, workdir=repo_root)

    sign_flip_payload = load_json(sign_flip_json)
    spectrum_payload = load_json(spectrum_json)
    propagation_payload = load_json(propagation_json)

    summary_text = build_third_model_summary_text(
        model_key=args.model_key,
        model_name_or_path=str(args.model_name_or_path),
        output_root=output_root,
        sign_flip_payload=sign_flip_payload,
        spectrum_payload=spectrum_payload,
        propagation_payload=propagation_payload,
    )
    summary_path = output_root / "summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(summary_text)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()

