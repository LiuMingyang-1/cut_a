"""Language-model inference helpers for building reusable R-Tuning caches."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cut_a_lab.prep.r_tuning.cache import write_inference_cache
from cut_a_lab.prep.r_tuning.contracts import InferenceSampleRecord, LayerCacheRecord, NormalizedSample


def require_transformers() -> Any:
    """Import transformers lazily so the rest of the package works without it."""
    try:
        import transformers
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency `transformers`. Install project dependencies before running the inference cache stage."
        ) from exc
    return transformers


def normalize_free_text(value: str) -> str:
    """Normalize free text for light-weight exact match checks."""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def classify_generation(sample: NormalizedSample, generated_text: str) -> tuple[int, int]:
    """Return (sample_label, silver_label) where 1 means hallucinated/incorrect."""
    predicted = str(generated_text).strip()
    expected = str(sample.expected_answer).strip()

    if sample.task_type == "multiple_choice_letter":
        predicted_label = predicted[:1].upper() if predicted else ""
        expected_label = expected[:1].upper()
        is_wrong = int(predicted_label != expected_label)
    elif sample.task_type == "classification_label":
        expected_norm = normalize_free_text(expected)
        choices = {normalize_free_text(choice): choice for choice in sample.choices}
        predicted_norm = normalize_free_text(predicted)
        if predicted_norm in choices:
            is_wrong = int(predicted_norm != expected_norm)
        else:
            is_wrong = 1
    else:
        is_wrong = int(normalize_free_text(predicted) != normalize_free_text(expected))

    return is_wrong, is_wrong


def _resolve_torch_device(device: str) -> str:
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def _find_final_norm_module(model: Any) -> Any | None:
    candidates = [
        getattr(getattr(model, "model", None), "norm", None),
        getattr(getattr(model, "transformer", None), "ln_f", None),
        getattr(getattr(model, "base_model", None), "norm", None),
        getattr(getattr(getattr(model, "base_model", None), "model", None), "norm", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if hasattr(candidate, "forward") or callable(candidate):
            return candidate
    return None


def _apply_final_norm(hidden: Any, final_norm: Any | None) -> Any:
    if final_norm is None:
        return hidden
    if hasattr(final_norm, "forward"):
        return final_norm(hidden)
    if callable(final_norm):
        return final_norm(hidden)
    return hidden


def _compute_entropy_from_logits(logits: Any) -> float:
    import torch

    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1)
    return float(entropy.mean().item())


@dataclass(frozen=True)
class ModelRunnerConfig:
    """Options for one inference-cache build run."""

    model_name_or_path: str
    device: str = "auto"
    max_new_tokens: int = 64
    trust_remote_code: bool = True


def _load_model_and_tokenizer(config: ModelRunnerConfig) -> tuple[Any, Any, str]:
    transformers = require_transformers()
    import torch

    resolved_device = _resolve_torch_device(config.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=torch.float16 if resolved_device in {"cuda", "mps"} else torch.float32,
    )
    model.to(resolved_device)
    model.eval()
    return model, tokenizer, resolved_device


def _generate_with_hidden_cache(
    *,
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    max_new_tokens: int,
    device: str,
) -> tuple[str, np.ndarray, np.ndarray, int]:
    import torch

    encoded = tokenizer(prompt_text, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_length = int(encoded["input_ids"].shape[1])

    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = generated[:, prompt_length:]
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    if generated_ids.shape[1] == 0:
        return generated_text, np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32), 0

    full_attention_mask = torch.ones_like(generated)
    with torch.no_grad():
        outputs = model(
            input_ids=generated,
            attention_mask=full_attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model forward pass did not return hidden states.")

    answer_slice = slice(prompt_length, generated.shape[1])
    final_norm = _find_final_norm_module(model)
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        raise RuntimeError("Model does not expose output embeddings; cannot compute per-layer entropy.")

    layer_means: list[np.ndarray] = []
    layer_entropy: list[float] = []
    for hidden in hidden_states:
        answer_hidden = hidden[:, answer_slice, :]
        if answer_hidden.shape[1] == 0:
            continue
        answer_hidden = answer_hidden.to(dtype=torch.float32)
        layer_means.append(answer_hidden.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32))

        normalized = _apply_final_norm(answer_hidden, final_norm)
        logits = output_embeddings(normalized)
        layer_entropy.append(_compute_entropy_from_logits(logits))

    return (
        generated_text,
        np.stack(layer_means, axis=0).astype(np.float32),
        np.asarray(layer_entropy, dtype=np.float32),
        int(generated_ids.shape[1]),
    )


def build_inference_cache(
    *,
    samples: list[NormalizedSample],
    output_dir: Path,
    config: ModelRunnerConfig,
) -> Path:
    """Run one model over normalized samples and persist a reusable cache."""
    model, tokenizer, resolved_device = _load_model_and_tokenizer(config)
    records: list[LayerCacheRecord] = []

    for sample in samples:
        generated_text, layer_hidden_mean, layer_entropy, answer_token_count = _generate_with_hidden_cache(
            model=model,
            tokenizer=tokenizer,
            prompt_text=sample.prompt_text,
            max_new_tokens=config.max_new_tokens,
            device=resolved_device,
        )

        if answer_token_count == 0:
            # Keep a minimal non-empty placeholder so downstream vector code remains valid.
            hidden_size = int(getattr(model.config, "hidden_size", 1) or 1)
            num_hidden_layers = int(getattr(model.config, "num_hidden_layers", 1) or 1)
            layer_count = max(2, num_hidden_layers + 1)
            layer_hidden_mean = np.zeros((layer_count, hidden_size), dtype=np.float32)
            layer_entropy = np.zeros((layer_count,), dtype=np.float32)

        sample_label, silver_label = classify_generation(sample, generated_text)
        cached_sample = InferenceSampleRecord(
            dataset_name=sample.dataset_name,
            split_name=sample.split_name,
            sample_id=sample.sample_id,
            span_id=f"{sample.sample_id}:full",
            prompt_text=sample.prompt_text,
            generated_text=generated_text,
            expected_answer=sample.expected_answer,
            sample_label=sample_label,
            silver_label=silver_label,
            task_type=sample.task_type,
            answer_token_count=answer_token_count,
            metadata=sample.metadata,
        )
        records.append(
            LayerCacheRecord(
                sample=cached_sample,
                layer_hidden_mean=layer_hidden_mean,
                layer_entropy=layer_entropy,
            )
        )

    manifest = {
        "model_name_or_path": config.model_name_or_path,
        "device": resolved_device,
        "max_new_tokens": config.max_new_tokens,
        "n_samples": len(records),
        "status": "ok",
    }
    write_inference_cache(output_dir=output_dir, records=records, manifest=manifest)
    return Path(output_dir)
