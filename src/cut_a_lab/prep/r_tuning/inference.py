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


def _extract_first_line(text: str) -> str:
    """Return the first non-empty line of a (possibly verbose) model output."""
    for line in str(text).split("\n"):
        line = line.strip()
        if line:
            return line
    return str(text).strip()


def classify_generation(sample: NormalizedSample, generated_text: str) -> tuple[int, int]:
    """Return (sample_label, silver_label) where 1 means hallucinated/incorrect."""
    predicted = str(generated_text).strip()
    expected = str(sample.expected_answer).strip()

    if sample.task_type == "multiple_choice_letter":
        predicted_label = predicted[:1].upper() if predicted else ""
        expected_label = expected[:1].upper()
        is_wrong = int(predicted_label != expected_label)
    elif sample.task_type == "classification_label":
        first_line = _extract_first_line(predicted)
        expected_norm = normalize_free_text(expected)
        choices = {normalize_free_text(choice): choice for choice in sample.choices}
        predicted_norm = normalize_free_text(first_line)
        if predicted_norm in choices:
            is_wrong = int(predicted_norm != expected_norm)
        else:
            is_wrong = 1
    else:
        first_line = _extract_first_line(predicted)
        expected_norm = normalize_free_text(expected)
        predicted_norm = normalize_free_text(first_line)
        if predicted_norm == expected_norm:
            is_wrong = 0
        elif expected_norm and expected_norm in predicted_norm:
            is_wrong = 0
        else:
            is_wrong = 1

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

    logits = logits.to(dtype=torch.float32)
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
    batch_size: int = 8


def _load_model_and_tokenizer(config: ModelRunnerConfig) -> tuple[Any, Any, str]:
    transformers = require_transformers()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    resolved_device = str(next(model.parameters()).device)
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
        # Keep model dtype for norm + logits computation; cast to float32 only for numpy storage
        layer_means.append(answer_hidden.mean(dim=1).squeeze(0).to(dtype=torch.float32).cpu().numpy().astype(np.float32))

        normalized = _apply_final_norm(answer_hidden, final_norm)
        logits = output_embeddings(normalized)
        layer_entropy.append(_compute_entropy_from_logits(logits))

    return (
        generated_text,
        np.stack(layer_means, axis=0).astype(np.float32),
        np.asarray(layer_entropy, dtype=np.float32),
        int(generated_ids.shape[1]),
    )


def _generate_batch_with_hidden_cache(
    *,
    model: Any,
    tokenizer: Any,
    prompt_texts: list[str],
    max_new_tokens: int,
    device: str,
) -> list[tuple[str, np.ndarray, np.ndarray, int]]:
    """Generate and extract hidden states for a batch of prompts.

    Generation is batched; hidden-state extraction runs per sample to avoid
    padding interference in the answer token slice.
    """
    import torch

    tokenizer.padding_side = "left"
    encoded = tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=False)
    encoded = {k: v.to(device) for k, v in encoded.items()}

    prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
    padded_len = int(encoded["input_ids"].shape[1])

    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    final_norm = _find_final_norm_module(model)
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        raise RuntimeError("Model does not expose output embeddings; cannot compute per-layer entropy.")

    results: list[tuple[str, np.ndarray, np.ndarray, int]] = []
    for i, prompt_len in enumerate(prompt_lengths):
        answer_ids = generated[i, padded_len:]

        # Find where actual answer ends (first eos after generation start)
        eos_positions = (answer_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        answer_end = int(eos_positions[0]) + 1 if len(eos_positions) > 0 else answer_ids.shape[0]
        answer_ids = answer_ids[:answer_end]
        answer_token_count = int(answer_ids.shape[0])

        generated_text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

        if answer_token_count == 0:
            results.append((generated_text, np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32), 0))
            continue

        # Reconstruct full sequence without left padding for hidden-state pass
        padding_amount = padded_len - int(prompt_len)
        prompt_ids = generated[i:i + 1, padding_amount:padded_len]
        full_seq = torch.cat([prompt_ids, answer_ids.unsqueeze(0)], dim=1)
        full_mask = torch.ones_like(full_seq)

        with torch.no_grad():
            outputs = model(
                input_ids=full_seq,
                attention_mask=full_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model forward pass did not return hidden states.")

        answer_slice = slice(int(prompt_len), full_seq.shape[1])
        layer_means: list[np.ndarray] = []
        layer_entropy_vals: list[float] = []
        for hidden in hidden_states:
            answer_hidden = hidden[:, answer_slice, :]
            if answer_hidden.shape[1] == 0:
                continue
            layer_means.append(answer_hidden.mean(dim=1).squeeze(0).to(dtype=torch.float32).cpu().numpy().astype(np.float32))
            normalized = _apply_final_norm(answer_hidden, final_norm)
            logits = output_embeddings(normalized)
            layer_entropy_vals.append(_compute_entropy_from_logits(logits))

        results.append((
            generated_text,
            np.stack(layer_means, axis=0).astype(np.float32),
            np.asarray(layer_entropy_vals, dtype=np.float32),
            answer_token_count,
        ))

    return results


def build_inference_cache(
    *,
    samples: list[NormalizedSample],
    output_dir: Path,
    config: ModelRunnerConfig,
    model: Any = None,
    tokenizer: Any = None,
    resolved_device: str | None = None,
) -> Path:
    """Run one model over normalized samples and persist a reusable cache.

    If *model*, *tokenizer*, and *resolved_device* are provided they are reused
    as-is (no reload).  Pass them when calling build_inference_cache in a loop
    to avoid reloading weights on every iteration.
    """
    if model is None or tokenizer is None or resolved_device is None:
        model, tokenizer, resolved_device = _load_model_and_tokenizer(config)
    records: list[LayerCacheRecord] = []

    batch_size = max(1, config.batch_size)
    hidden_size = int(getattr(model.config, "hidden_size", 1) or 1)
    num_hidden_layers = int(getattr(model.config, "num_hidden_layers", 1) or 1)
    placeholder_layer_count = max(2, num_hidden_layers + 1)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    batches = [samples[i: i + batch_size] for i in range(0, len(samples), batch_size)]
    iterator = tqdm(batches, desc="inference", unit="batch", total=len(batches)) if tqdm else batches
    for batch in iterator:
        batch_results = _generate_batch_with_hidden_cache(
            model=model,
            tokenizer=tokenizer,
            prompt_texts=[s.prompt_text for s in batch],
            max_new_tokens=config.max_new_tokens,
            device=resolved_device,
        )

        for sample, (generated_text, layer_hidden_mean, layer_entropy, answer_token_count) in zip(batch, batch_results):
            if answer_token_count == 0:
                layer_hidden_mean = np.zeros((placeholder_layer_count, hidden_size), dtype=np.float32)
                layer_entropy = np.zeros((placeholder_layer_count,), dtype=np.float32)

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
        "batch_size": batch_size,
        "n_samples": len(records),
        "status": "ok",
    }
    write_inference_cache(output_dir=output_dir, records=records, manifest=manifest)
    return Path(output_dir)
