"""In-memory and streaming checkpoint inference."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..text import normalize_notes
from .checkpoints import validate_checkpoint_labels


def load_checkpoint(model_dir: Path, device: torch.device | None = None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    labels = validate_checkpoint_labels(model)
    return tokenizer, model, labels, device


def predict_texts(
    model_dir: Path,
    texts: list[str],
    *,
    batch_size: int = 8,
    max_length: int = 512,
    device: torch.device | None = None,
) -> tuple[list[str], np.ndarray]:
    tokenizer, model, labels, device = load_checkpoint(model_dir, device)
    normalized = [normalize_notes(text) for text in texts]
    if any(not text for text in normalized):
        raise ValueError("Cannot predict empty notes")
    logits_chunks = []
    with torch.inference_mode():
        for start in range(0, len(normalized), batch_size):
            encoded = tokenizer(
                normalized[start:start + batch_size], truncation=True,
                max_length=max_length, padding=True, return_tensors="pt",
            ).to(device)
            logits_chunks.append(model(**encoded).logits.float().cpu().numpy())
    logits = np.concatenate(logits_chunks) if logits_chunks else np.empty((0, len(labels)))
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
    batch_size: int = 8,
    max_length: int = 512,
    read_chunk_size: int = 512,
    limit: int | None = None,
    progress: Callable[[int], None] | None = None,
) -> int:
    """Stream a large event CSV without retaining all notes or embeddings."""
    tokenizer, model, labels, device = load_checkpoint(model_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    total = 0
    wrote_header = False
    for chunk in pd.read_csv(
        events_path, usecols=["event_id_cnty", "notes"],
        chunksize=read_chunk_size, low_memory=False,
    ):
        if limit is not None and total >= limit:
            break
        if limit is not None:
            chunk = chunk.iloc[:max(0, limit - total)]
        texts = chunk.notes.map(normalize_notes).tolist()
        if any(not text for text in texts):
            raise ValueError("Event source contains empty notes")
        predicted: list[str] = []
        confidence: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                encoded = tokenizer(
                    texts[start:start + batch_size], truncation=True,
                    max_length=max_length, padding=True, return_tensors="pt",
                ).to(device)
                logits = model(**encoded).logits.float()
                probabilities = torch.softmax(logits, dim=-1)
                scores, indices = probabilities.max(dim=-1)
                predicted.extend(labels[int(index)] for index in indices.cpu())
                confidence.extend(scores.cpu().tolist())
                del encoded, logits, probabilities, scores, indices
        result = pd.DataFrame({
            "event_id_cnty": chunk.event_id_cnty.to_numpy(),
            "predicted_label": predicted,
            "confidence": np.round(confidence, 6),
        })
        result.to_csv(output_path, mode="a", header=not wrote_header, index=False)
        wrote_header = True
        total += len(result)
        if progress:
            progress(total)
        del chunk, texts, predicted, confidence, result
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    return total
