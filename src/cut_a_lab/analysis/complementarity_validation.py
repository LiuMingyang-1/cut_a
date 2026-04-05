"""Validate whether single-model disagreement translates into combined-model gains."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cut_a_lab.analysis.disagreement_analysis import QUADRANT_LABELS, classify_span_quadrants
from cut_a_lab.core.io import dump_json, read_jsonl


def _load_predictions(path: Path, *, threshold: float) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(Path(path))
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not row.get("is_labeled") or row.get("probability") is None:
            continue
        probability = float(row["probability"])
        silver_label = int(row["silver_label"])
        prediction = int(probability >= threshold)
        result[row["span_id"]] = {
            "span_id": row["span_id"],
            "sample_id": str(row["sample_id"]),
            "silver_label": silver_label,
            "probability": probability,
            "prediction": prediction,
            "correct": prediction == silver_label,
        }
    return result


def _accuracy(entries: list[dict[str, Any]]) -> float:
    if not entries:
        return 0.0
    return sum(1 for entry in entries if entry["correct"]) / len(entries)


def summarize_model_on_groups(
    groups: dict[str, list[dict[str, Any]]],
    predictions_by_span: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Summarize a model's behavior on disagreement groups defined by icr vs entropy."""
    result: dict[str, Any] = {}
    for group_key, entries in groups.items():
        rows = [predictions_by_span[entry["span_id"]] for entry in entries if entry["span_id"] in predictions_by_span]
        result[group_key] = {
            "n": len(rows),
            "accuracy": _accuracy(rows),
            "n_correct": sum(1 for row in rows if row["correct"]),
            "n_wrong": sum(1 for row in rows if not row["correct"]),
        }
    return result


def summarize_against_entropy(
    entropy_predictions: dict[str, dict[str, Any]],
    target_predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Measure rescue vs regression relative to entropy-only."""
    shared_span_ids = sorted(set(entropy_predictions) & set(target_predictions))
    rescue = 0
    regression = 0
    both_correct = 0
    both_wrong = 0

    for span_id in shared_span_ids:
        ent_correct = entropy_predictions[span_id]["correct"]
        target_correct = target_predictions[span_id]["correct"]
        if not ent_correct and target_correct:
            rescue += 1
        elif ent_correct and not target_correct:
            regression += 1
        elif ent_correct and target_correct:
            both_correct += 1
        else:
            both_wrong += 1

    total = len(shared_span_ids)
    return {
        "n_shared": total,
        "rescue_count": rescue,
        "regression_count": regression,
        "net_gain_count": rescue - regression,
        "rescue_rate_over_entropy_errors": rescue / max(1, rescue + both_wrong),
        "regression_rate_over_entropy_correct": regression / max(1, regression + both_correct),
        "net_gain_rate_over_all": (rescue - regression) / max(1, total),
    }


def summarize_group_rescue(
    groups: dict[str, list[dict[str, Any]]],
    entropy_predictions: dict[str, dict[str, Any]],
    target_predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Measure rescue/regression relative to entropy-only within each disagreement group."""
    result: dict[str, Any] = {}
    for group_key, entries in groups.items():
        rescue = 0
        regression = 0
        both_correct = 0
        both_wrong = 0
        total = 0

        for entry in entries:
            span_id = entry["span_id"]
            if span_id not in entropy_predictions or span_id not in target_predictions:
                continue
            total += 1
            ent_correct = entropy_predictions[span_id]["correct"]
            target_correct = target_predictions[span_id]["correct"]
            if not ent_correct and target_correct:
                rescue += 1
            elif ent_correct and not target_correct:
                regression += 1
            elif ent_correct and target_correct:
                both_correct += 1
            else:
                both_wrong += 1

        result[group_key] = {
            "n": total,
            "rescue_count": rescue,
            "regression_count": regression,
            "net_gain_count": rescue - regression,
            "rescue_rate_over_entropy_errors": rescue / max(1, rescue + both_wrong),
            "regression_rate_over_entropy_correct": regression / max(1, regression + both_correct),
            "target_accuracy": (rescue + both_correct) / max(1, total),
        }
    return result


def run_complementarity_validation(
    *,
    icr_oof_path: Path,
    entropy_oof_path: Path,
    target_model_paths: dict[str, Path],
    output_dir: Path,
    threshold: float = 0.5,
) -> dict[str, Any]:
    icr_oof_path = Path(icr_oof_path)
    entropy_oof_path = Path(entropy_oof_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Classifying disagreement groups...")
    groups = classify_span_quadrants(icr_oof_path, entropy_oof_path, threshold=threshold)
    counts = {group_key: len(entries) for group_key, entries in groups.items()}

    print("Loading baseline predictions...")
    icr_predictions = _load_predictions(icr_oof_path, threshold=threshold)
    entropy_predictions = _load_predictions(entropy_oof_path, threshold=threshold)

    report: dict[str, Any] = {
        "threshold": threshold,
        "quadrant_counts": counts,
        "models": {},
    }

    for model_name, path in target_model_paths.items():
        print(f"Evaluating {model_name}...")
        target_predictions = _load_predictions(path, threshold=threshold)
        report["models"][model_name] = {
            "path": str(path),
            "group_accuracy": summarize_model_on_groups(groups, target_predictions),
            "vs_entropy_overall": summarize_against_entropy(entropy_predictions, target_predictions),
            "vs_entropy_by_group": summarize_group_rescue(groups, entropy_predictions, target_predictions),
            "vs_icr_by_group": summarize_group_rescue(groups, icr_predictions, target_predictions),
        }

    dump_json(output_dir / "complementarity_validation_report.json", report)

    summary_lines = [
        f"Complementarity validation (threshold={threshold})",
        "",
        "Quadrant counts:",
    ]
    for group_key in ("group_a", "group_b", "group_c", "group_d"):
        summary_lines.append(f"  {QUADRANT_LABELS[group_key]}: {counts[group_key]}")

    summary_lines.extend(["", "Combined-model behavior relative to entropy-only:"])
    for model_name, payload in report["models"].items():
        overall = payload["vs_entropy_overall"]
        by_group = payload["vs_entropy_by_group"]
        summary_lines.append(
            f"  {model_name}: "
            f"net_gain={overall['net_gain_count']} "
            f"(rescue={overall['rescue_count']}, regression={overall['regression_count']}) "
            f"net_gain_rate={overall['net_gain_rate_over_all']:.4f}"
        )
        summary_lines.append(
            f"    group_b rescue={by_group['group_b']['rescue_count']}/{counts['group_b']} "
            f"({by_group['group_b']['rescue_rate_over_entropy_errors']:.3f}), "
            f"group_a regression={by_group['group_a']['regression_count']}/{counts['group_a']} "
            f"({by_group['group_a']['regression_rate_over_entropy_correct']:.3f})"
        )
        summary_lines.append(
            f"    accuracies: "
            f"A={payload['group_accuracy']['group_a']['accuracy']:.3f} "
            f"B={payload['group_accuracy']['group_b']['accuracy']:.3f} "
            f"C={payload['group_accuracy']['group_c']['accuracy']:.3f} "
            f"D={payload['group_accuracy']['group_d']['accuracy']:.3f}"
        )

    summary_text = "\n".join(summary_lines) + "\n"
    (output_dir / "complementarity_validation_summary.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text)

    return report
