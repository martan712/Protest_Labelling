"""Run a saved sequence-classification checkpoint on a locked manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# The finetuner package lives in the repository-root pipeline/ directory,
# which is three levels up from evaluation/scoring/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "pipeline"))
from finetuner.data import build_input  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", default="evaluation/manifests/dev_manifest.csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    frame = pd.read_csv(args.manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device).eval()
    texts = [build_input(row) for _, row in frame.iterrows()]
    predictions = []
    with torch.no_grad():
        for start in range(0, len(texts), args.batch_size):
            encoded = tokenizer(texts[start:start + args.batch_size], truncation=True, max_length=128, padding=True, return_tensors="pt").to(device)
            predictions.extend(model(**encoded).logits.argmax(dim=-1).cpu().tolist())
    labels = model.config.id2label
    frame["predicted_class"] = [labels.get(str(i), labels.get(i, str(i))) for i in predictions]
    frame[["event_id_cnty", "predicted_class"]].to_csv(args.out, index=False)
    print(f"wrote {len(frame)} predictions to {args.out}")


if __name__ == "__main__":
    main()
