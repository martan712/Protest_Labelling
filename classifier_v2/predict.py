"""Label every filtered ACLED event with a saved ModernBERT checkpoint.

The input is only the original ``notes`` string — no country, year, actor,
``clean_notes``, old label or keyword is passed to the model.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from classifier_v2.common import CLASS_NAMES, DATA_DIR, label_maps, normalize_notes, reject_empty_notes

DEFAULT_EVENTS = DATA_DIR / "filtered_events.csv"
DEFAULT_OUT = DATA_DIR / "classifier_v2" / "all_events_predictions.csv"


def predict_all(
    model_dir: Path,
    events_path: Path = DEFAULT_EVENTS,
    out_path: Path = DEFAULT_OUT,
    *,
    batch_size: int = 8,
    max_len: int = 512,
    read_chunksize: int = 512,
    limit: int | None = None,
) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    id2label, _ = label_maps()
    resolved = {int(key): value for key, value in model.config.id2label.items()}
    if resolved != id2label:
        raise ValueError(f"Checkpoint label mapping differs from taxonomy: {resolved}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Prediction output is a derived artefact.  Start a fresh file so a
    # restarted run cannot silently append duplicate rows to a partial result.
    out_path.unlink(missing_ok=True)
    total = 0
    wrote_header = False
    for chunk in pd.read_csv(
        events_path,
        usecols=["event_id_cnty", "notes"],
        chunksize=read_chunksize,
        low_memory=False,
    ):
        if limit is not None and total >= limit:
            break
        if limit is not None:
            chunk = chunk.iloc[: max(0, limit - total)]
        chunk = chunk.copy()
        chunk["notes"] = chunk["notes"].map(normalize_notes)
        reject_empty_notes(chunk, context=str(events_path))
        texts = chunk["notes"].tolist()
        predictions: list[str] = []
        confidences: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                encoded = tokenizer(
                    texts[start:start + batch_size],
                    truncation=True,
                    max_length=max_len,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                logits = model(**encoded).logits.float()
                probabilities = torch.softmax(logits, dim=-1)
                confidence, indices = probabilities.max(dim=-1)
                predictions.extend(id2label[int(index)] for index in indices.cpu())
                confidences.extend(confidence.cpu().tolist())
                # Keep peak GPU memory bounded on long full-corpus runs.
                del encoded, logits, probabilities, confidence, indices
        result = pd.DataFrame({
            "event_id_cnty": chunk["event_id_cnty"].to_numpy(),
            "predicted_label": predictions,
            "confidence": np.round(confidences, 6),
        })
        result.to_csv(out_path, mode="a", header=not wrote_header, index=False)
        wrote_header = True
        total += len(result)
        print(f"predicted {total} rows", flush=True)
        del chunk, texts, predictions, confidences, result
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    print(f"wrote {total} predictions to {out_path}")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Inference batch size; keep small for bounded GPU memory.")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="Label only the first N events (smoke test).")
    args = parser.parse_args()
    predict_all(
        args.model, args.events, args.out,
        batch_size=args.batch_size, max_len=args.max_len, limit=args.limit,
    )


if __name__ == "__main__":
    main()
