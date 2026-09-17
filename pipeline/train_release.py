"""Non-interactive 21-class release training entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from finetuner.data import load_data, build_dataloaders
from finetuner.model import get_device, load_tokenizer, train_model, evaluate_model, save_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/release/labeled_balanced_21.csv")
    parser.add_argument("--model", default="models/attempt7-t5-leave-keywords-in-epoch-2/hf_transformer_model")
    parser.add_argument("--output", default="models/release21-t5")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--freeze-encoder", action="store_true")
    args = parser.parse_args()
    data_path = Path(args.data)
    if args.max_rows:
        full = pd.read_csv(data_path)
        labels = full["final_label"]
        counts = labels.value_counts().sort_index()
        quota = (counts / counts.sum() * args.max_rows).round().astype(int).clip(lower=1)
        while quota.sum() > args.max_rows:
            largest = quota[quota > 1].idxmax()
            quota.loc[largest] -= 1
        while quota.sum() < args.max_rows:
            quota.loc[counts.idxmax()] += 1
        sampled = pd.concat([
            group.sample(n=int(quota.loc[label]), random_state=42)
            for label, group in full.groupby("final_label", sort=True)
        ]).sample(frac=1, random_state=42)
        subset = data_path.parent / "_training_subset_21.csv"
        sampled.to_csv(subset, index=False)
        data_path = subset
    splits = load_data(data_path, train_split=0.8, val_split=0.1, random_state=42)
    if len(splits.class_names) != 21 or "other" not in splits.class_names:
        raise ValueError(f"Expected 21 labels including other, got {splits.class_names}")
    device, device_name = get_device()
    tokenizer = load_tokenizer(args.model)
    loaders = build_dataloaders(splits, tokenizer, args.max_len, args.batch_size)
    model, train_losses, val_losses, training_time = train_model(
        args.model, splits.class_names, loaders[0], loaders[1], device,
        learning_rate=5e-5, epochs=args.epochs, output_dir=args.output, tokenizer=tokenizer,
        freeze_encoder=args.freeze_encoder,
    )
    save_model(model, tokenizer, args.output)
    metrics = {
        "model": args.model, "output": args.output, "device": device_name,
        "epochs": args.epochs, "batch_size": args.batch_size, "max_len": args.max_len,
        "freeze_encoder": args.freeze_encoder,
        "training_rows": len(splits.train), "validation_rows": len(splits.val), "test_rows": len(splits.test),
        "class_names": splits.class_names, "train_losses": train_losses, "val_losses": val_losses,
        "training_time_seconds": training_time,
        "validation": evaluate_model(model, loaders[1], device),
        "test": evaluate_model(model, loaders[2], device),
    }
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "release21_metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n")
    print(json.dumps(metrics, indent=2, default=float))


if __name__ == "__main__":
    main()
