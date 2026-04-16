"""Unify ParaRel experiments onto the full OOD protocol and rebuild the main table.

This script is an orchestration wrapper with conservative defaults:
1. Keep legacy roots untouched (especially old llama outputs).
2. Rebuild/patch only the minimum chain needed for full-OOD consistency.
3. Emit a single markdown report with:
   - unified main table
   - reusable vs deprecated outputs
   - artifact paths for citation

Example:
    uv run python scripts/experiments/unify_pararel_full_ood.py \
        --llama-model-name-or-path /root/autodl-tmp/hf/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
        --mistral-model-name-or-path /root/autodl-tmp/hf/models/mistralai/Mistral-7B-Instruct-v0.3 \
        --allow-local-7b
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cut_a_lab.prep.r_tuning.datasets import discover_available_dataset_splits, load_normalized_samples

PARAREL_SPLITS = ("train", "id_test", "ood_test")
EVAL_SPLITS = ("id_test", "ood_test")
REQUIRED_MAIN_TABLE_METHODS = (
    "train-label sign (oracle)",
    "train-sign transfer",
    "pseudo-sign (train infer)",
    "pseudo-sign (on-split infer)",
    "agreement rate (direct)",
)


@dataclass(frozen=True)
class ModelArtifacts:
    model_name: str
    root: Path
    inference_manifest_ood: Path
    method_input_combined_ood: Path
    self_consistency_summary: Path
    agreement_summary: Path


@dataclass(frozen=True)
class ModelMetrics:
    model_name: str
    root: Path
    ood_n_samples_manifest: int
    ood_n_samples_method_input: int
    ood_n_samples_self_consistency: int
    ood_n_samples_agreement: int
    methods: dict[str, tuple[float, float]]
    feature_spectrum_ood_n: int | None
    sign_flip_ood_n: int | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/R-Tuning-data"))
    parser.add_argument(
        "--llama-source-root",
        type=Path,
        default=Path("outputs/experiments/llama-3.1-8b-instruct"),
        help="Read-only source root for legacy llama artifacts.",
    )
    parser.add_argument(
        "--llama-target-root",
        type=Path,
        default=Path("outputs/experiments/llama-3.1-8b-instruct-fullood"),
        help="New llama root for full-OOD chain rebuild.",
    )
    parser.add_argument(
        "--mistral-source-root",
        type=Path,
        default=Path("outputs/experiments/mistral-7b-instruct"),
    )
    parser.add_argument(
        "--mistral-target-root",
        type=Path,
        default=Path("outputs/experiments/mistral-7b-instruct"),
        help="Can be same as source root, or a new full-OOD root.",
    )
    parser.add_argument("--llama-model-name-or-path", default=None)
    parser.add_argument("--mistral-model-name-or-path", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prompt-batch-size", type=int, default=4)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-local-7b", action="store_true")
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Re-run steps even when full-OOD artifacts already exist.",
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--run-optional-analyses", action="store_true")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("outputs/experiments/comparison/pararel_fullood_main_table.md"),
    )
    return parser.parse_args()


def _run(cmd: list[str], *, workdir: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=workdir, check=True)


def _copy_tree_if_missing(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    if not src.exists():
        raise FileNotFoundError(f"Missing source dir to bootstrap: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    print(f"[copy] {src} -> {dst}")


def _copy_file_if_missing(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[copy] {src} -> {dst}")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _count_jsonl_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _discover_pararel_split_counts(data_root: Path) -> dict[str, int]:
    specs = discover_available_dataset_splits(data_root)
    pararel_specs = {spec.split_name: spec for spec in specs if spec.dataset_name == "pararel"}
    missing = [split_name for split_name in PARAREL_SPLITS if split_name not in pararel_specs]
    if missing:
        raise FileNotFoundError(f"Missing ParaRel data splits under {data_root}: {missing}")
    counts: dict[str, int] = {}
    for split_name in PARAREL_SPLITS:
        rows = load_normalized_samples(root_dir=data_root, spec=pararel_specs[split_name])
        counts[split_name] = len(rows)
    return counts


def _resolve_model_name(
    preferred: str | None,
    *,
    primary_manifest: Path,
    fallback_manifest: Path | None = None,
) -> str:
    if preferred:
        return preferred
    candidates = [primary_manifest]
    if fallback_manifest is not None:
        candidates.append(fallback_manifest)
    for path in candidates:
        if not path.exists():
            continue
        payload = _load_json(path)
        value = payload.get("model_name_or_path")
        if value:
            return str(value)
    raise FileNotFoundError(
        "Could not resolve model_name_or_path from manifests. "
        "Pass --llama-model-name-or-path / --mistral-model-name-or-path explicitly."
    )


def _manifest_matches_full_ood(manifest_path: Path, expected_ood_n: int) -> bool:
    if not manifest_path.exists():
        return False
    payload = _load_json(manifest_path)
    n_samples = int(payload.get("n_samples", -1))
    if n_samples != expected_ood_n:
        return False
    # Some older manifests don't carry subset metadata; count match is still acceptable.
    subset_applied = payload.get("subset_applied")
    if subset_applied is None:
        return True
    return bool(subset_applied) is False


def _parse_metric_token(token: str) -> float:
    return float("nan") if token == "nan" else float(token)


def _parse_two_column_method_table(summary_text: str) -> dict[str, tuple[float, float]]:
    rows: dict[str, tuple[float, float]] = {}
    pattern = re.compile(
        r"^(?P<name>.+?)\s+(?P<id>nan|-?\d+\.\d+)\s+(?P<ood>nan|-?\d+\.\d+)\s*$"
    )
    in_table = False
    for raw_line in summary_text.splitlines():
        line = raw_line.rstrip()
        if not in_table:
            if line.startswith("method"):
                in_table = True
            continue
        if not line.strip():
            break
        match = pattern.match(line)
        if not match:
            continue
        rows[match.group("name").strip()] = (
            _parse_metric_token(match.group("id")),
            _parse_metric_token(match.group("ood")),
        )
    return rows


def _parse_split_sizes_from_self_summary(summary_text: str) -> dict[str, int]:
    sizes: dict[str, int] = {}
    pattern = re.compile(r"^\[(?P<split>[a-z_]+)\]\s+n=(?P<n>\d+)\b")
    for raw_line in summary_text.splitlines():
        match = pattern.match(raw_line.strip())
        if not match:
            continue
        sizes[match.group("split")] = int(match.group("n"))
    return sizes


def _parse_split_sizes_from_agreement_summary(summary_text: str) -> dict[str, int]:
    sizes: dict[str, int] = {}
    pattern = re.compile(r"^(?P<split>train|id_test|ood_test)\s+AUROC=\d+\.\d+\s+\(n=(?P<n>\d+)")
    for raw_line in summary_text.splitlines():
        match = pattern.match(raw_line.strip())
        if not match:
            continue
        sizes[match.group("split")] = int(match.group("n"))
    return sizes


def _parse_ood_n_from_feature_spectrum(path: Path) -> int | None:
    if not path.exists():
        return None
    payload = _load_json(path)
    splits = payload.get("splits", {})
    ood = splits.get("ood_test", {})
    value = ood.get("n_samples")
    return None if value is None else int(value)


def _parse_ood_n_from_sign_flip(path: Path) -> int | None:
    if not path.exists():
        return None
    payload = _load_json(path)
    splits = payload.get("splits", {})
    ood = splits.get("ood_test", {})
    value = ood.get("n_samples")
    return None if value is None else int(value)


def _ensure_inference_ood_full(
    *,
    repo_root: Path,
    data_root: Path,
    model_root: Path,
    model_name_or_path: str,
    device: str,
    max_new_tokens: int,
    batch_size: int,
    allow_local_7b: bool,
    expected_ood_n: int,
    skip_existing: bool,
    check_only: bool,
) -> None:
    manifest = model_root / "inference" / "pararel" / "ood_test" / "manifest.json"
    if skip_existing and _manifest_matches_full_ood(manifest, expected_ood_n):
        return
    if check_only:
        raise FileNotFoundError(f"Expected full-OOD manifest missing/mismatch: {manifest}")

    cmd = [
        sys.executable,
        "scripts/build_r_tuning_inference_cache.py",
        "--data-root",
        str(data_root),
        "--output-root",
        str(model_root / "inference"),
        "--model-name-or-path",
        model_name_or_path,
        "--device",
        device,
        "--max-new-tokens",
        str(max_new_tokens),
        "--batch-size",
        str(batch_size),
        "--dataset",
        "pararel",
        "--split",
        "ood_test",
    ]
    if allow_local_7b:
        cmd.append("--allow-local-7b")
    _run(cmd, workdir=repo_root)

    if not _manifest_matches_full_ood(manifest, expected_ood_n):
        raise RuntimeError(
            f"OOD manifest is not full after rebuild: {manifest}. "
            f"Expected n_samples={expected_ood_n} with subset_applied=false."
        )


def _ensure_method_inputs(
    *,
    repo_root: Path,
    data_root: Path,
    model_root: Path,
    expected_counts: dict[str, int],
    skip_existing: bool,
    check_only: bool,
) -> None:
    def _all_ready() -> bool:
        for split_name, expected_n in expected_counts.items():
            combined_path = model_root / "method_inputs" / "pararel" / split_name / "combined_spans.jsonl"
            if not combined_path.exists():
                return False
            if _count_jsonl_lines(combined_path) != expected_n:
                return False
        return True

    if skip_existing and _all_ready():
        return
    if check_only:
        raise FileNotFoundError(f"Method-input chain incomplete under: {model_root / 'method_inputs' / 'pararel'}")

    cmd = [
        sys.executable,
        "scripts/build_r_tuning_method_inputs.py",
        "--data-root",
        str(data_root),
        "--cache-root",
        str(model_root / "inference"),
        "--output-root",
        str(model_root / "method_inputs"),
        "--dataset",
        "pararel",
    ]
    for split_name in PARAREL_SPLITS:
        cmd.extend(["--split", split_name])
    _run(cmd, workdir=repo_root)

    if not _all_ready():
        raise RuntimeError(f"Method-input rebuild did not produce expected counts at {model_root}.")


def _ensure_self_consistency_and_agreement(
    *,
    repo_root: Path,
    model_root: Path,
    device: str,
    prompt_batch_size: int,
    k: int,
    temperature: float,
    top_p: float,
    seed: int,
    expected_ood_n: int,
    skip_existing: bool,
    check_only: bool,
) -> None:
    self_summary = model_root / "self_consistency" / "summary.txt"
    agreement_summary = model_root / "self_consistency" / "agreement_baseline_summary.txt"

    def _self_is_ready() -> bool:
        if not self_summary.exists():
            return False
        sizes = _parse_split_sizes_from_self_summary(self_summary.read_text(encoding="utf-8"))
        return sizes.get("ood_test") == expected_ood_n

    def _agreement_is_ready() -> bool:
        if not agreement_summary.exists():
            return False
        text = agreement_summary.read_text(encoding="utf-8")
        sizes = _parse_split_sizes_from_agreement_summary(text)
        rows = _parse_two_column_method_table(text)
        return sizes.get("ood_test") == expected_ood_n and "agreement rate (direct)" in rows

    if not (skip_existing and _self_is_ready()):
        if check_only:
            raise FileNotFoundError(f"Self-consistency summary missing/mismatch: {self_summary}")
        cmd = [
            sys.executable,
            "scripts/experiments/self_consistency_sign.py",
            "--model-output-root",
            str(model_root),
            "--device",
            device,
            "--splits",
            *PARAREL_SPLITS,
            "--eval-splits",
            *EVAL_SPLITS,
            "--k",
            str(k),
            "--temperature",
            str(temperature),
            "--top-p",
            str(top_p),
            "--prompt-batch-size",
            str(prompt_batch_size),
            "--seed",
            str(seed),
            "--reuse-existing-samples",
        ]
        _run(cmd, workdir=repo_root)

    if not (skip_existing and _agreement_is_ready()):
        if check_only:
            raise FileNotFoundError(f"Agreement summary missing/mismatch: {agreement_summary}")
        cmd = [
            sys.executable,
            "scripts/experiments/agreement_baseline.py",
            "--model-output-root",
            str(model_root),
            "--splits",
            *PARAREL_SPLITS,
            "--eval-splits",
            *EVAL_SPLITS,
        ]
        _run(cmd, workdir=repo_root)

    if not _self_is_ready():
        raise RuntimeError(f"Self-consistency summary does not reflect full OOD at: {self_summary}")
    if not _agreement_is_ready():
        raise RuntimeError(f"Agreement summary does not reflect full OOD at: {agreement_summary}")


def _maybe_run_optional_analyses(
    *,
    repo_root: Path,
    model_root: Path,
    expected_ood_n: int,
    run_optional_analyses: bool,
    check_only: bool,
) -> None:
    if not run_optional_analyses:
        return
    method_input_root = model_root / "method_inputs" / "pararel"
    sign_flip_json = model_root / "sign_flip_validation" / "sign_flip_validation.json"
    feature_json = model_root / "feature_correctness_spectrum" / "feature_correctness_spectrum.json"

    sign_flip_n = _parse_ood_n_from_sign_flip(sign_flip_json)
    feature_n = _parse_ood_n_from_feature_spectrum(feature_json)
    sign_flip_ready = sign_flip_n == expected_ood_n
    feature_ready = feature_n == expected_ood_n
    if sign_flip_ready and feature_ready:
        return
    if check_only:
        raise FileNotFoundError("Optional analyses are not full-OOD aligned in check-only mode.")

    if not sign_flip_ready:
        _run(
            [
                sys.executable,
                "scripts/experiments/validate_sign_flip.py",
                "--method-input-root",
                str(method_input_root),
                "--output-dir",
                str(model_root / "sign_flip_validation"),
            ],
            workdir=repo_root,
        )
    if not feature_ready:
        _run(
            [
                sys.executable,
                "scripts/experiments/feature_correctness_spectrum.py",
                "--method-input-root",
                str(method_input_root),
                "--output-dir",
                str(model_root / "feature_correctness_spectrum"),
            ],
            workdir=repo_root,
        )


def _bootstrap_llama_target(source_root: Path, target_root: Path) -> None:
    if source_root.resolve() == target_root.resolve():
        raise ValueError(
            "llama-source-root and llama-target-root must be different to avoid overriding legacy llama outputs."
        )
    for split_name in ("train", "id_test"):
        _copy_tree_if_missing(
            source_root / "inference" / "pararel" / split_name,
            target_root / "inference" / "pararel" / split_name,
        )
        _copy_file_if_missing(
            source_root / "self_consistency" / f"samples_{split_name}.jsonl",
            target_root / "self_consistency" / f"samples_{split_name}.jsonl",
        )


def _bootstrap_mistral_target(source_root: Path, target_root: Path) -> None:
    if source_root.resolve() == target_root.resolve():
        return
    _copy_tree_if_missing(source_root / "inference" / "pararel", target_root / "inference" / "pararel")
    _copy_tree_if_missing(source_root / "method_inputs" / "pararel", target_root / "method_inputs" / "pararel")
    _copy_tree_if_missing(
        source_root / "feature_correctness_spectrum",
        target_root / "feature_correctness_spectrum",
    )
    _copy_tree_if_missing(source_root / "sign_flip_validation", target_root / "sign_flip_validation")
    for split_name in ("train", "id_test"):
        _copy_file_if_missing(
            source_root / "self_consistency" / f"samples_{split_name}.jsonl",
            target_root / "self_consistency" / f"samples_{split_name}.jsonl",
        )


def _collect_model_metrics(
    *,
    model_name: str,
    root: Path,
    expected_ood_n: int,
) -> ModelMetrics:
    artifacts = ModelArtifacts(
        model_name=model_name,
        root=root,
        inference_manifest_ood=root / "inference" / "pararel" / "ood_test" / "manifest.json",
        method_input_combined_ood=root / "method_inputs" / "pararel" / "ood_test" / "combined_spans.jsonl",
        self_consistency_summary=root / "self_consistency" / "summary.txt",
        agreement_summary=root / "self_consistency" / "agreement_baseline_summary.txt",
    )
    for path in (
        artifacts.inference_manifest_ood,
        artifacts.method_input_combined_ood,
        artifacts.self_consistency_summary,
        artifacts.agreement_summary,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact for {model_name}: {path}")

    manifest = _load_json(artifacts.inference_manifest_ood)
    ood_n_manifest = int(manifest["n_samples"])
    if ood_n_manifest != expected_ood_n:
        raise RuntimeError(
            f"{model_name}: inference manifest n_samples={ood_n_manifest}, expected full OOD {expected_ood_n}."
        )
    ood_n_method_input = _count_jsonl_lines(artifacts.method_input_combined_ood)
    if ood_n_method_input != expected_ood_n:
        raise RuntimeError(
            f"{model_name}: method_inputs ood_test rows={ood_n_method_input}, expected {expected_ood_n}."
        )

    self_text = artifacts.self_consistency_summary.read_text(encoding="utf-8")
    agreement_text = artifacts.agreement_summary.read_text(encoding="utf-8")
    self_rows = _parse_two_column_method_table(self_text)
    agreement_rows = _parse_two_column_method_table(agreement_text)

    self_sizes = _parse_split_sizes_from_self_summary(self_text)
    agreement_sizes = _parse_split_sizes_from_agreement_summary(agreement_text)
    ood_n_self = int(self_sizes.get("ood_test", -1))
    ood_n_agreement = int(agreement_sizes.get("ood_test", -1))
    if ood_n_self != expected_ood_n or ood_n_agreement != expected_ood_n:
        raise RuntimeError(
            f"{model_name}: self/agreement summaries are not full OOD "
            f"(self={ood_n_self}, agreement={ood_n_agreement}, expected={expected_ood_n})."
        )

    methods: dict[str, tuple[float, float]] = {}
    for method_name in REQUIRED_MAIN_TABLE_METHODS:
        if method_name == "agreement rate (direct)":
            if method_name not in agreement_rows:
                raise KeyError(f"{model_name}: missing method row in agreement summary: {method_name}")
            methods[method_name] = agreement_rows[method_name]
            continue
        if method_name not in self_rows:
            raise KeyError(f"{model_name}: missing method row in self-consistency summary: {method_name}")
        methods[method_name] = self_rows[method_name]

    feature_ood_n = _parse_ood_n_from_feature_spectrum(root / "feature_correctness_spectrum" / "feature_correctness_spectrum.json")
    sign_flip_ood_n = _parse_ood_n_from_sign_flip(root / "sign_flip_validation" / "sign_flip_validation.json")

    return ModelMetrics(
        model_name=model_name,
        root=root,
        ood_n_samples_manifest=ood_n_manifest,
        ood_n_samples_method_input=ood_n_method_input,
        ood_n_samples_self_consistency=ood_n_self,
        ood_n_samples_agreement=ood_n_agreement,
        methods=methods,
        feature_spectrum_ood_n=feature_ood_n,
        sign_flip_ood_n=sign_flip_ood_n,
    )


def _fmt_metric(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.4f}"


def _build_main_table(metrics: list[ModelMetrics]) -> str:
    lines = [
        "| model | method | id_test AUROC | ood_test AUROC | ood_test n_samples |",
        "|---|---|---:|---:|---:|",
    ]
    for model_metrics in metrics:
        for method_name in REQUIRED_MAIN_TABLE_METHODS:
            id_value, ood_value = model_metrics.methods[method_name]
            lines.append(
                "| "
                + " | ".join(
                    [
                        model_metrics.model_name,
                        method_name,
                        _fmt_metric(id_value),
                        _fmt_metric(ood_value),
                        str(model_metrics.ood_n_samples_manifest),
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def _build_artifact_table(metrics: list[ModelMetrics]) -> str:
    lines = [
        "| model | root | inference manifest | method_inputs ood combined | self_consistency summary | agreement summary |",
        "|---|---|---|---|---|---|",
    ]
    for model_metrics in metrics:
        root = model_metrics.root
        lines.append(
            "| "
            + " | ".join(
                [
                    model_metrics.model_name,
                    str(root),
                    str(root / "inference" / "pararel" / "ood_test" / "manifest.json"),
                    str(root / "method_inputs" / "pararel" / "ood_test" / "combined_spans.jsonl"),
                    str(root / "self_consistency" / "summary.txt"),
                    str(root / "self_consistency" / "agreement_baseline_summary.txt"),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _build_optional_checks_table(metrics: list[ModelMetrics], expected_ood_n: int) -> str:
    lines = [
        "| model | feature_correctness_spectrum ood n | sign_flip_validation ood n | full-OOD aligned? |",
        "|---|---:|---:|---|",
    ]
    for model_metrics in metrics:
        feature_n = model_metrics.feature_spectrum_ood_n
        sign_flip_n = model_metrics.sign_flip_ood_n
        aligned = feature_n == expected_ood_n and sign_flip_n == expected_ood_n
        lines.append(
            "| "
            + " | ".join(
                [
                    model_metrics.model_name,
                    "missing" if feature_n is None else str(feature_n),
                    "missing" if sign_flip_n is None else str(sign_flip_n),
                    "yes" if aligned else "no",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _build_report_markdown(
    *,
    expected_counts: dict[str, int],
    llama_metrics: ModelMetrics,
    mistral_metrics: ModelMetrics,
    llama_source_root: Path,
    llama_target_root: Path,
    mistral_source_root: Path,
    mistral_target_root: Path,
    ood5584_root: Path,
) -> str:
    expected_ood_n = expected_counts["ood_test"]
    metrics = [llama_metrics, mistral_metrics]
    lines = [
        "# ParaRel full OOD unified report",
        "",
        f"- full OOD target count: **{expected_ood_n}**",
        f"- train/id counts: train={expected_counts['train']}, id_test={expected_counts['id_test']}",
        "",
        "## Main table (full OOD)",
        _build_main_table(metrics),
        "",
        "## Result statement",
        f"1. Llama full OOD补齐状态：**已补齐**，`ood_test n_samples={llama_metrics.ood_n_samples_manifest}`。",
        (
            "2. Mistral主链是否已是full OOD：**是**，且用于主表的self-consistency/agreement链路"
            f"为 `ood_test n_samples={mistral_metrics.ood_n_samples_manifest}`。"
        ),
        (
            "3. 失效旧结果：`outputs/experiments/mistral-7b-instruct-ood5584` 为 paired subset，"
            "不能作为 full OOD 主表依据。"
        ),
        (
            "4. 可复用结果：Mistral 在 `"
            f"{mistral_target_root}` 下的 full OOD inference/method_inputs 可复用；"
            "Llama legacy root 仅可作历史对照。"
        ),
        (
            "5. 论文主表引用路径：Llama 用 `"
            f"{llama_target_root}`，Mistral 用 `{mistral_target_root}`。"
        ),
        "",
        "## Artifact paths",
        _build_artifact_table(metrics),
        "",
        "## Optional full-OOD analysis checks",
        _build_optional_checks_table(metrics, expected_ood_n),
        "",
        "## Legacy/source references",
        f"- llama source root (read-only): `{llama_source_root}`",
        f"- llama full-OOD target root: `{llama_target_root}`",
        f"- mistral source root: `{mistral_source_root}`",
        f"- mistral active root: `{mistral_target_root}`",
        f"- paired subset root (deprecated for main table): `{ood5584_root}`",
        "",
    ]
    return "\n".join(lines)


def _write_report(report_path: Path, markdown_text: str, payload: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(markdown_text, encoding="utf-8")
    report_json_path = report_path.with_suffix(".json")
    report_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved report: {report_path}")
    print(f"Saved report json: {report_json_path}")


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    skip_existing = not args.force_rebuild
    expected_counts = _discover_pararel_split_counts(args.data_root)
    expected_ood_n = expected_counts["ood_test"]

    if not args.check_only:
        _bootstrap_llama_target(args.llama_source_root, args.llama_target_root)
        _bootstrap_mistral_target(args.mistral_source_root, args.mistral_target_root)

    llama_model_name = _resolve_model_name(
        args.llama_model_name_or_path,
        primary_manifest=args.llama_target_root / "inference" / "pararel" / "train" / "manifest.json",
        fallback_manifest=args.llama_source_root / "inference" / "pararel" / "train" / "manifest.json",
    )
    mistral_model_name = _resolve_model_name(
        args.mistral_model_name_or_path,
        primary_manifest=args.mistral_target_root / "inference" / "pararel" / "train" / "manifest.json",
        fallback_manifest=args.mistral_source_root / "inference" / "pararel" / "train" / "manifest.json",
    )

    _ensure_inference_ood_full(
        repo_root=repo_root,
        data_root=args.data_root,
        model_root=args.llama_target_root,
        model_name_or_path=llama_model_name,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        allow_local_7b=args.allow_local_7b,
        expected_ood_n=expected_ood_n,
        skip_existing=skip_existing,
        check_only=args.check_only,
    )
    _ensure_method_inputs(
        repo_root=repo_root,
        data_root=args.data_root,
        model_root=args.llama_target_root,
        expected_counts=expected_counts,
        skip_existing=skip_existing,
        check_only=args.check_only,
    )
    _ensure_self_consistency_and_agreement(
        repo_root=repo_root,
        model_root=args.llama_target_root,
        device=args.device,
        prompt_batch_size=args.prompt_batch_size,
        k=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        expected_ood_n=expected_ood_n,
        skip_existing=skip_existing,
        check_only=args.check_only,
    )
    _maybe_run_optional_analyses(
        repo_root=repo_root,
        model_root=args.llama_target_root,
        expected_ood_n=expected_ood_n,
        run_optional_analyses=args.run_optional_analyses,
        check_only=args.check_only,
    )

    _ensure_inference_ood_full(
        repo_root=repo_root,
        data_root=args.data_root,
        model_root=args.mistral_target_root,
        model_name_or_path=mistral_model_name,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        allow_local_7b=args.allow_local_7b,
        expected_ood_n=expected_ood_n,
        skip_existing=skip_existing,
        check_only=args.check_only,
    )
    _ensure_method_inputs(
        repo_root=repo_root,
        data_root=args.data_root,
        model_root=args.mistral_target_root,
        expected_counts=expected_counts,
        skip_existing=skip_existing,
        check_only=args.check_only,
    )
    _ensure_self_consistency_and_agreement(
        repo_root=repo_root,
        model_root=args.mistral_target_root,
        device=args.device,
        prompt_batch_size=args.prompt_batch_size,
        k=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        expected_ood_n=expected_ood_n,
        skip_existing=skip_existing,
        check_only=args.check_only,
    )
    _maybe_run_optional_analyses(
        repo_root=repo_root,
        model_root=args.mistral_target_root,
        expected_ood_n=expected_ood_n,
        run_optional_analyses=args.run_optional_analyses,
        check_only=args.check_only,
    )

    llama_metrics = _collect_model_metrics(
        model_name="llama-3.1-8b-instruct",
        root=args.llama_target_root,
        expected_ood_n=expected_ood_n,
    )
    mistral_metrics = _collect_model_metrics(
        model_name="mistral-7b-instruct",
        root=args.mistral_target_root,
        expected_ood_n=expected_ood_n,
    )

    report_markdown = _build_report_markdown(
        expected_counts=expected_counts,
        llama_metrics=llama_metrics,
        mistral_metrics=mistral_metrics,
        llama_source_root=args.llama_source_root,
        llama_target_root=args.llama_target_root,
        mistral_source_root=args.mistral_source_root,
        mistral_target_root=args.mistral_target_root,
        ood5584_root=Path("outputs/experiments/mistral-7b-instruct-ood5584"),
    )

    report_payload = {
        "expected_counts": expected_counts,
        "llama": {
            "root": str(llama_metrics.root),
            "ood_n_samples_manifest": llama_metrics.ood_n_samples_manifest,
            "ood_n_samples_method_input": llama_metrics.ood_n_samples_method_input,
            "ood_n_samples_self_consistency": llama_metrics.ood_n_samples_self_consistency,
            "ood_n_samples_agreement": llama_metrics.ood_n_samples_agreement,
            "methods": llama_metrics.methods,
            "feature_spectrum_ood_n": llama_metrics.feature_spectrum_ood_n,
            "sign_flip_ood_n": llama_metrics.sign_flip_ood_n,
        },
        "mistral": {
            "root": str(mistral_metrics.root),
            "ood_n_samples_manifest": mistral_metrics.ood_n_samples_manifest,
            "ood_n_samples_method_input": mistral_metrics.ood_n_samples_method_input,
            "ood_n_samples_self_consistency": mistral_metrics.ood_n_samples_self_consistency,
            "ood_n_samples_agreement": mistral_metrics.ood_n_samples_agreement,
            "methods": mistral_metrics.methods,
            "feature_spectrum_ood_n": mistral_metrics.feature_spectrum_ood_n,
            "sign_flip_ood_n": mistral_metrics.sign_flip_ood_n,
        },
        "paths": {
            "report_markdown": str(args.report_path),
            "report_json": str(args.report_path.with_suffix(".json")),
        },
    }
    _write_report(args.report_path, report_markdown, report_payload)
    print(report_markdown)


if __name__ == "__main__":
    main()
