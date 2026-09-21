"""Length-bucketed, mixed-precision checkpoint inference."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from ..text import normalize_notes
from .checkpoints import validate_checkpoint_labels

DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_LENGTH = 256
DEFAULT_READ_CHUNK_SIZE = 1024


def inference_dtype(device: torch.device, *, use_bfloat16: bool = True) -> torch.dtype:
    if (
        use_bfloat16
        and device.type == "cuda"
        and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    ):
        return torch.bfloat16
    return torch.float32


def load_checkpoint(
    model_dir: Path,
    device: torch.device | None = None,
    *,
    use_bfloat16: bool = True,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = inference_dtype(device, use_bfloat16=use_bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, dtype=dtype)
    model = model.to(device).eval()
    labels = validate_checkpoint_labels(model)
    return tokenizer, model, labels, device, dtype


def _predict_loaded(
    tokenizer,
    model,
    texts: list[str],
    *,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> np.ndarray:
    """Predict one bounded chunk, grouping similar lengths and restoring order."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not texts:
        return np.empty((0, model.config.num_labels), dtype=np.float32)
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )
    order = sorted(range(len(texts)), key=lambda index: len(encoded["input_ids"][index]))
    logits = np.empty((len(texts), model.config.num_labels), dtype=np.float32)
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            features = [
                {key: values[index] for key, values in encoded.items()}
                for index in indices
            ]
            batch = collator(features).to(device)
            batch_logits = model(**batch).logits.float().cpu().numpy()
            logits[indices] = batch_logits
            del batch, batch_logits, features
    return logits


def predict_texts(
    model_dir: Path,
    texts: list[str],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
    device: torch.device | None = None,
    use_bfloat16: bool = True,
) -> tuple[list[str], np.ndarray]:
    tokenizer, model, labels, device, _ = load_checkpoint(
        model_dir, device, use_bfloat16=use_bfloat16
    )
    normalized = [normalize_notes(text) for text in texts]
    if any(not text for text in normalized):
        raise ValueError("Cannot predict empty notes")
    logits = _predict_loaded(
        tokenizer, model, normalized,
        batch_size=batch_size, max_length=max_length, device=device,
    )
    predictions = [labels[int(index)] for index in logits.argmax(axis=-1)]
    del model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions, logits


def predict_csv(
    model_dir: Path,
    events_path: Path,
    output_path: Path,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
    read_chunk_size: int = DEFAULT_READ_CHUNK_SIZE,
    limit: int | None = None,
    progress: Callable[[int], None] | None = None,
    use_bfloat16: bool = True,
) -> int:
    """Stream a large event CSV without retaining all notes or embeddings."""
    tokenizer, model, labels, device, _ = load_checkpoint(
        model_dir, use_bfloat16=use_bfloat16
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    total = 0
    wrote_header = False
    for chunk in pd.read_csv(
        events_path,
        usecols=["event_id_cnty", "notes"],
        chunksize=read_chunk_size,
        low_memory=False,
    ):
        if limit is not None and total >= limit:
            break
        if limit is not None:
            chunk = chunk.iloc[:max(0, limit - total)]
        texts = chunk.notes.map(normalize_notes).tolist()
        if any(not text for text in texts):
            raise ValueError("Event source contains empty notes")
        logits = _predict_loaded(
            tokenizer, model, texts,
            batch_size=batch_size, max_length=max_length, device=device,
        )
        probabilities = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
        indices = probabilities.argmax(axis=-1)
        confidence = probabilities[np.arange(len(probabilities)), indices]
        result = pd.DataFrame({
            "event_id_cnty": chunk.event_id_cnty.to_numpy(),
            "predicted_label": [labels[int(index)] for index in indices],
            "confidence": np.round(confidence, 6),
        })
        result.to_csv(output_path, mode="a", header=not wrote_header, index=False)
        wrote_header = True
        total += len(result)
        if progress:
            progress(total)
        del chunk, texts, logits, probabilities, indices, confidence, result
    del model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return total
