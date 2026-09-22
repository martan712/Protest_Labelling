#!/usr/bin/env python3
"""Train one nested release for one or more random seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protest_classifier.data.training_sets import TrainingRelease, load_training_release
from protest_classifier.modeling.training import TrainingConfig, train

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=int, choices=[800, 1500, 3000, 6000], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--dev-csv", type=Path, default=ROOT / "data/annotations/dev.csv")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "artifacts/models/new_classifier")
    parser.add_argument("--model", default="answerdotai/ModernBERT-base")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument(
        "--length-grouping", action=argparse.BooleanOptionalAction, default=True,
        help="Group similar token lengths to avoid padding waste (default: enabled).",
    )
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True,
        help="Recompute activations to cap training memory (default: enabled).",
    )
    args = parser.parse_args()
    train_path = args.train_csv or ROOT / f"data/annotations/train_{args.release:04d}.csv"
    release = load_training_release(train_path, args.dev_csv)
    release = TrainingRelease(f"run-{args.release}", release.train, release.dev)
    config = TrainingConfig(
        model_name=args.model, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, epochs=args.epochs,
        batch_size=args.batch_size, gradient_accumulation=args.grad_accum,
        max_length=args.max_length, patience=args.patience,
        group_by_length=args.length_grouping,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    for seed in args.seeds:
        output = args.models_dir / f"run-{args.release}" / f"seed-{seed}"
        metrics = train(release, output, seed=seed, config=config)
        summary = {
            key: metrics[key] for key in (
                "run", "seed", "train_rows", "dev_rows", "best_epoch",
                "best_dev_macro_f1", "best_dev_accuracy", "truncated_train_notes",
            )
        }
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
