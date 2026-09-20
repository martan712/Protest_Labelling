"""Evaluate ModernBERT checkpoints on development or the locked test set.

``--split dev``  → ``data/classifier_v2/dev_497.csv`` (model selection).
``--split test`` → the locked 840-row set.  This is the *only* entry point that
reads ``data/review/test_gold_v1/test_gold.csv``; it is hard-coded here and is
never exposed as an argument, so training code cannot reach it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from classifier_v2.common import (
    CLASS_NAMES,
    DEV_GOLD,
    MODELS_DIR,
    RESULTS_DIR,
    load_dev_497,
    load_test_set,
    label_maps,
    normalize_notes,
)
from evaluation.taxonomy import accepted

DEFAULT_MAX_LEN = 512
DEFAULT_BATCH_SIZE = 32

# Legacy reference numbers on the identical 840 IDs (unweighted strict accuracy).
LEGACY_COMPARISON = {
    "students_t5": 0.701,
    "release21": 0.702,
}


def resolve_model_dirs(model: Path | None, runs: list[str] | None, split: str) -> list[Path]:
    dirs: list[Path] = []
    if model is not None:
        dirs.append(Path(model))
    for run in runs or []:
        run_dir = MODELS_DIR / run
        if not run_dir.exists():
            raise SystemExit(f"Run directory not found: {run_dir}")
        for seed_dir in sorted(run_dir.glob("seed-*")):
            candidate = seed_dir / "best"
            dirs.append(candidate if candidate.exists() else seed_dir)
    if not dirs:
        raise SystemExit("Pass --model and/or --run")
    for directory in dirs:
        if not directory.exists():
            raise SystemExit(f"Model directory not found: {directory}")
    # Stable, de-duplicated order.
    seen: set[str] = set()
    unique = []
    for directory in dirs:
        key = str(directory)
        if key not in seen:
            seen.add(key)
            unique.append(directory)
    return unique


def predict_frame(
    model_dir: Path,
    frame: pd.DataFrame,
    *,
    max_len: int = DEFAULT_MAX_LEN,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: torch.device | None = None,
) -> tuple[list[str], np.ndarray]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()

    model_id2label = model.config.id2label
    canonical_id2label, _ = label_maps()
    # Inference must use the same mapping as training.
    resolved = {
        int(key): (value if isinstance(value, str) else value)
        for key, value in model_id2label.items()
    }
    if resolved != canonical_id2label:
        raise ValueError(
            "Model label mapping differs from evaluation.taxonomy.CLASS_NAMES; "
            f"checkpoint={resolved}"
        )

    texts = [normalize_notes(text) for text in frame["notes"]]
    logits_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start:start + batch_size],
                truncation=True,
                max_length=max_len,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits_chunks.append(model(**encoded).logits.float().cpu().numpy())
    logits = np.concatenate(logits_chunks, axis=0) if logits_chunks else np.zeros((0, len(CLASS_NAMES)))
    predictions = [canonical_id2label[int(index)] for index in logits.argmax(axis=-1)]
    return predictions, logits


def _split_alternatives(value: object) -> tuple[str, ...]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ()
    return tuple(part for part in str(value).split("|") if part)


def score_predictions(frame: pd.DataFrame, predictions: list[str]) -> dict:
    from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

    truth = frame["primary_label"].tolist()
    accuracy = float(np.mean([p == t for p, t in zip(predictions, truth)]))
    macro_f1 = float(f1_score(truth, predictions, labels=CLASS_NAMES, average="macro", zero_division=0))
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, labels=CLASS_NAMES, zero_division=0
    )
    per_class = {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(CLASS_NAMES)
    }
    matrix = confusion_matrix(truth, predictions, labels=CLASS_NAMES)
    accepted_hits = [
        accepted(prediction, primary, _split_alternatives(alternatives))
        for prediction, primary, alternatives in zip(
            predictions, frame["primary_label"], frame["alternative_labels"]
        )
    ]
    other_index = CLASS_NAMES.index("other")
    other_predicted = int(sum(prediction == "other" for prediction in predictions))
    other_support = int(sum(label == "other" for label in truth))
    return {
        "n": len(frame),
        "strict_accuracy": accuracy,
        "macro_f1": macro_f1,
        "accepted_accuracy": float(np.mean(accepted_hits)),
        "per_class": per_class,
        "labels": list(CLASS_NAMES),
        "confusion_matrix": matrix.tolist(),
        "other": {
            "predicted": other_predicted,
            "support": other_support,
            "recall": per_class["other"]["recall"],
        },
    }


def _mean_range(model_results: list[dict]) -> dict:
    def summary(key: str) -> dict:
        values = [result[key] for result in model_results]
        return {"mean": float(np.mean(values)), "min": float(np.min(values)), "max": float(np.max(values))}

    return {"strict_accuracy": summary("strict_accuracy"), "macro_f1": summary("macro_f1")}


def evaluate(
    model_dirs: list[Path],
    split: str,
    *,
    max_len: int = DEFAULT_MAX_LEN,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
    if split == "dev":
        frame = load_dev_497()
    elif split == "test":
        frame = load_test_set()
    else:
        raise ValueError("split must be 'dev' or 'test'")

    results = []
    for model_dir in model_dirs:
        predictions, _ = predict_frame(model_dir, frame, max_len=max_len, batch_size=batch_size)
        metrics = score_predictions(frame, predictions)
        seed = None
        run_name = None
        for part in model_dir.parts:
            if part.startswith("seed-"):
                seed = part.replace("seed-", "")
            if part.startswith("run-"):
                run_name = part
        results.append({"model": str(model_dir), "run": run_name, "seed": seed, **metrics})

    report: dict = {
        "split": split,
        "split_file": str(DEV_GOLD if split == "dev" else "evaluation/manifests/test_manifest_840.csv + test_gold.csv"),
        "model_name": "answerdotai/ModernBERT-base",
        "max_len": max_len,
        "models": results,
    }
    if len(results) > 1:
        report["mean"] = _mean_range(results)
        by_run: dict[str, list[dict]] = {}
        for result in results:
            by_run.setdefault(result.get("run") or "model", []).append(result)
        report["by_run"] = {
            name: {**{r["seed"]: r["strict_accuracy"] for r in group}, "summary": _mean_range(group)}
            for name, group in by_run.items()
        }
    if split == "test":
        strict = [result["strict_accuracy"] for result in results]
        report["comparison"] = {
            **LEGACY_COMPARISON,
            "new_modernbert": float(np.mean(strict)),
            "note": "Unweighted strict accuracy on the same 840 IDs.",
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--run", nargs="+", default=None,
                        help="Run name(s) under models/classifier_v2 (all seeds of each).")
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.split == "test" and not args.run and args.model is None:
        raise SystemExit("Pass --run or --model to score the test set")
    model_dirs = resolve_model_dirs(args.model, args.run, args.split)
    report = evaluate(model_dirs, args.split, max_len=args.max_len, batch_size=args.batch_size)

    out = args.out or RESULTS_DIR / f"classifier_v2_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")
    for result in report["models"]:
        print(
            f"  seed={result['seed']} strict={result['strict_accuracy']:.4f} "
            f"macro_f1={result['macro_f1']:.4f} accepted={result['accepted_accuracy']:.4f} "
            f"other_pred={result['other']['predicted']}"
        )


if __name__ == "__main__":
    main()
