"""Dataset adapters for local R-Tuning raw data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from cut_a_lab.prep.r_tuning.contracts import NormalizedSample


@dataclass(frozen=True)
class DatasetFileSpec:
    """Location and adapter for one dataset split."""

    dataset_name: str
    split_name: str
    relative_path: str
    adapter_name: str


def _normalize_text(value: str) -> str:
    return " ".join(str(value).strip().split())


def _join_sentences(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(str(part).strip() for part in value if str(part).strip())
    return str(value)


def _render_hotpot_context(context: list[list[Any]]) -> str:
    parts: list[str] = []
    for title, sentences in context:
        joined = " ".join(str(sentence).strip() for sentence in sentences if str(sentence).strip())
        parts.append(f"{title}: {joined}")
    return "\n".join(parts)


def _render_wice_evidence(evidence: list[str]) -> str:
    lines = [str(item).strip() for item in evidence if str(item).strip()]
    return "\n".join(lines)


def _load_json_like(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows


def _normalize_halueval(records: Iterable[dict[str, Any]], *, dataset_name: str, split_name: str) -> list[NormalizedSample]:
    samples: list[NormalizedSample] = []
    for index, row in enumerate(records):
        prompt = (
            "Answer the question using only the provided knowledge.\n\n"
            f"Knowledge:\n{_normalize_text(row['knowledge'])}\n\n"
            f"Question:\n{_normalize_text(row['question'])}\n\n"
            "Answer:"
        )
        samples.append(
            NormalizedSample(
                dataset_name=dataset_name,
                split_name=split_name,
                sample_id=f"{dataset_name}:{split_name}:{index}",
                prompt_text=prompt,
                expected_answer=_normalize_text(row["right_answer"]),
                task_type="qa_exact_match",
                metadata={
                    "question": row["question"],
                    "knowledge": row["knowledge"],
                    "hallucinated_answer": row.get("hallucinated_answer"),
                },
            )
        )
    return samples


def _normalize_hotpot(records: Iterable[dict[str, Any]], *, dataset_name: str, split_name: str) -> list[NormalizedSample]:
    samples: list[NormalizedSample] = []
    for index, row in enumerate(records):
        prompt = (
            "Read the context and answer the question concisely.\n\n"
            f"Context:\n{_render_hotpot_context(row['context'])}\n\n"
            f"Question:\n{_normalize_text(row['question'])}\n\n"
            "Answer:"
        )
        sample_id = row.get("_id", f"{dataset_name}:{split_name}:{index}")
        samples.append(
            NormalizedSample(
                dataset_name=dataset_name,
                split_name=split_name,
                sample_id=str(sample_id),
                prompt_text=prompt,
                expected_answer=_normalize_text(row["answer"]),
                task_type="qa_exact_match",
                metadata={
                    "question": row["question"],
                    "level": row.get("level"),
                    "type": row.get("type"),
                    "supporting_facts": row.get("supporting_facts"),
                },
            )
        )
    return samples


def _normalize_fever(records: Iterable[dict[str, Any]], *, dataset_name: str, split_name: str) -> list[NormalizedSample]:
    samples: list[NormalizedSample] = []
    choices = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
    for index, row in enumerate(records):
        prompt = (
            "Determine whether the evidence supports the claim.\n"
            "Reply with exactly one label: SUPPORTS, REFUTES, or NOT ENOUGH INFO.\n\n"
            f"Claim:\n{_normalize_text(row['claim'])}\n\n"
            f"Evidence:\n{_join_sentences(row['evidence'])}\n\n"
            "Label:"
        )
        samples.append(
            NormalizedSample(
                dataset_name=dataset_name,
                split_name=split_name,
                sample_id=f"{dataset_name}:{split_name}:{index}",
                prompt_text=prompt,
                expected_answer=str(row["label"]).strip(),
                task_type="classification_label",
                choices=choices,
                metadata={"claim": row["claim"], "evidence": row["evidence"]},
            )
        )
    return samples


def _normalize_wice(records: Iterable[dict[str, Any]], *, dataset_name: str, split_name: str) -> list[NormalizedSample]:
    samples: list[NormalizedSample] = []
    choices = ("supported", "partially_supported", "not_supported")
    for index, row in enumerate(records):
        prompt = (
            "Determine whether the evidence supports the claim.\n"
            "Reply with exactly one label: supported, partially_supported, or not_supported.\n\n"
            f"Claim:\n{_normalize_text(row['claim'])}\n\n"
            f"Evidence:\n{_render_wice_evidence(row['evidence'])}\n\n"
            "Label:"
        )
        samples.append(
            NormalizedSample(
                dataset_name=dataset_name,
                split_name=split_name,
                sample_id=f"{dataset_name}:{split_name}:{index}",
                prompt_text=prompt,
                expected_answer=str(row["label"]).strip(),
                task_type="classification_label",
                choices=choices,
                metadata={
                    "claim": row["claim"],
                    "supporting_sentences": row.get("supporting_sentences"),
                },
            )
        )
    return samples


def _normalize_mmlu(records: dict[str, list[list[str]]], *, dataset_name: str, split_name: str) -> list[NormalizedSample]:
    samples: list[NormalizedSample] = []
    for subject, questions in records.items():
        for index, row in enumerate(questions):
            question, option_a, option_b, option_c, option_d, answer = row
            prompt = (
                "Answer the multiple-choice question.\n"
                "Reply with exactly one option letter: A, B, C, or D.\n\n"
                f"Subject: {subject}\n"
                f"Question: {_normalize_text(question)}\n"
                f"A. {_normalize_text(option_a)}\n"
                f"B. {_normalize_text(option_b)}\n"
                f"C. {_normalize_text(option_c)}\n"
                f"D. {_normalize_text(option_d)}\n\n"
                "Answer:"
            )
            sample_id = f"{dataset_name}:{split_name}:{subject}:{index}"
            samples.append(
                NormalizedSample(
                    dataset_name=dataset_name,
                    split_name=split_name,
                    sample_id=sample_id,
                    prompt_text=prompt,
                    expected_answer=str(answer).strip(),
                    task_type="multiple_choice_letter",
                    choices=("A", "B", "C", "D"),
                    metadata={"subject": subject, "question": question},
                )
            )
    return samples


def _normalize_pararel(records: list[list[str]], *, dataset_name: str, split_name: str) -> list[NormalizedSample]:
    samples: list[NormalizedSample] = []
    for index, row in enumerate(records):
        question, answer, relation = row
        prompt = f"{_normalize_text(question)}\nAnswer:"
        samples.append(
            NormalizedSample(
                dataset_name=dataset_name,
                split_name=split_name,
                sample_id=f"{dataset_name}:{split_name}:{index}",
                prompt_text=prompt,
                expected_answer=_normalize_text(answer),
                task_type="qa_exact_match",
                metadata={"question": question, "relation": relation},
            )
        )
    return samples


ADAPTERS: dict[str, Callable[[Any], list[NormalizedSample]]] = {
    "halueval": lambda records: _normalize_halueval(records, dataset_name="HaluEvalQA", split_name="default"),
    "hotpot": lambda records: _normalize_hotpot(records, dataset_name="HotpotQA", split_name="default"),
    "fever": lambda records: _normalize_fever(records, dataset_name="FEVER", split_name="default"),
    "wice": lambda records: _normalize_wice(records, dataset_name="WiCE", split_name="default"),
    "mmlu": lambda records: _normalize_mmlu(records, dataset_name="MMLU", split_name="default"),
    "pararel": lambda records: _normalize_pararel(records, dataset_name="pararel", split_name="default"),
}


DATASET_FILE_SPECS: tuple[DatasetFileSpec, ...] = (
    DatasetFileSpec("HaluEvalQA", "default", "HaluEvalQA/HaluEvalQA.json", "halueval"),
    DatasetFileSpec("HotpotQA", "train", "HotpotQA/hotpot_10k.json", "hotpot"),
    DatasetFileSpec("HotpotQA", "test", "HotpotQA/hotpot_test.json", "hotpot"),
    DatasetFileSpec("FEVER", "train", "FEVER/fever_10k.json", "fever"),
    DatasetFileSpec("FEVER", "test", "FEVER/fever_10k_test.json", "fever"),
    DatasetFileSpec("WiCE", "train", "WiCE/wice_train.json", "wice"),
    DatasetFileSpec("WiCE", "test", "WiCE/wice_test.json", "wice"),
    DatasetFileSpec("MMLU", "id_train", "MMLU/MMLU_ID_train.json", "mmlu"),
    DatasetFileSpec("MMLU", "id_test", "MMLU/MMLU_ID_test.json", "mmlu"),
    DatasetFileSpec("MMLU", "ood_test", "MMLU/MMLU_OOD_test.json", "mmlu"),
    DatasetFileSpec("pararel", "train", "pararel/training_data.json", "pararel"),
    DatasetFileSpec("pararel", "id_test", "pararel/ID_test_pararel.json", "pararel"),
    DatasetFileSpec("pararel", "ood_test", "pararel/OOD_test_pararel.json", "pararel"),
)


def discover_available_dataset_splits(root_dir: Path) -> list[DatasetFileSpec]:
    """Return dataset splits that exist under the local R-Tuning root."""
    root = Path(root_dir)
    available: list[DatasetFileSpec] = []
    for spec in DATASET_FILE_SPECS:
        if (root / spec.relative_path).exists():
            available.append(spec)
    return available


def _adapter_for_name(name: str) -> Callable[[Any], list[NormalizedSample]]:
    if name not in ADAPTERS:
        raise KeyError(f"Unknown dataset adapter {name!r}. Available: {', '.join(sorted(ADAPTERS))}")
    return ADAPTERS[name]


def load_normalized_samples(*, root_dir: Path, spec: DatasetFileSpec) -> list[NormalizedSample]:
    """Load and normalize one dataset split from the local R-Tuning root."""
    path = Path(root_dir) / spec.relative_path
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    payload = _load_json_like(path)
    adapter = _adapter_for_name(spec.adapter_name)
    samples = adapter(payload)
    rewritten: list[NormalizedSample] = []
    for sample in samples:
        rewritten.append(
            NormalizedSample(
                dataset_name=spec.dataset_name,
                split_name=spec.split_name,
                sample_id=sample.sample_id,
                prompt_text=sample.prompt_text,
                expected_answer=sample.expected_answer,
                task_type=sample.task_type,
                choices=sample.choices,
                metadata=sample.metadata,
            )
        )
    return rewritten
