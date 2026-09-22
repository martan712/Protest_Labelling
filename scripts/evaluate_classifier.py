#!/usr/bin/env python3
"""Evaluate checkpoints on development or the separately locked test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from protest_classifier.data.training_sets import load_annotated
from protest_classifier.evaluation.metrics import examples, score
from protest_classifier.evaluation.test_data import load_locked_test
from protest_classifier.modeling.checkpoints import find_checkpoints
from protest_classifier.modeling.inference import predict_texts

ROOT = Path(__file__).resolve().parents[1]


def identity(model_dir: Path) -> tuple[str | None, str | None]:
    run = next((part for part in model_dir.parts if part.startswith("run-")), None)
    seed_part = next((part for part in model_dir.parts if part.startswith("seed-")), None)
    return run, seed_part.removeprefix("seed-") if seed_part else None


def mean_range(results: list[dict], key: str) -> dict:
    values = [result[key] for result in results]
    return {"mean": float(np.mean(values)), "min": float(np.min(values)), "max": float(np.max(values))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--run", nargs="+", default=[])
    parser.add_argument("--model", type=Path, action="append", default=[])
    parser.add_argument("--models-dir", type=Path, default=ROOT / "artifacts/models/new_classifier")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--bf16", action=argparse.BooleanOptionalAction, default=True,
        help="Use BF16 on supported GPUs (default: enabled).",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    model_dirs = [*args.model, *find_checkpoints(args.models_dir, args.run)]
    if not model_dirs:
        raise SystemExit("Pass --run and/or --model")
    if args.split == "dev":
        frame = load_annotated(ROOT / "data/annotations/dev.csv", context="development")
        source = "data/annotations/dev.csv"
    else:
        frame = load_locked_test(
            ROOT / "data/manifests/test_locked.csv",
            ROOT / "data/annotations/test_locked.csv",
        )
        source = "data/manifests/test_locked.csv + data/annotations/test_locked.csv"
    results = []
    for model_dir in model_dirs:
        predictions, _ = predict_texts(
            model_dir, frame.notes.tolist(),
            batch_size=args.batch_size, max_length=args.max_length,
            use_bfloat16=args.bf16,
        )
        run, seed = identity(model_dir)
        metrics = score(frame, predictions)
        metrics["examples"] = examples(frame, predictions)
        results.append({"model": str(model_dir), "run": run, "seed": seed, **metrics})
    report = {
        "split": args.split, "split_file": source,
        "max_length": args.max_length, "bf16": args.bf16, "models": results,
    }
    if len(results) > 1:
        report["mean"] = {
            key: mean_range(results, key)
            for key in ("strict_accuracy", "macro_f1", "accepted_accuracy")
        }
    output = args.output or ROOT / f"artifacts/reports/new_classifier_{args.split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {output}")
    for result in results:
        print(
            f"{result['run']} seed={result['seed']} "
            f"strict={result['strict_accuracy']:.4f} "
            f"macro_f1={result['macro_f1']:.4f} "
            f"accepted={result['accepted_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
