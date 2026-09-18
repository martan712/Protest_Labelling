"""Non-interactive 21-class release training entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from finetuner.data import CLASS_NAMES, load_data, build_dataloaders
from finetuner.model import get_device, load_tokenizer, train_model, evaluate_model, save_model
from finetuner.data import build_input
from evaluation.taxonomy import accepted


def validate_training_data(data_path, heldout_path):
    frame = pd.read_csv(data_path, low_memory=False)
    required = {"event_id_cnty", "final_label", "duplicate_group_id", "note_hash", "event_date"}
    if required - set(frame):
        raise ValueError(f"Training data is missing columns: {sorted(required - set(frame))}")
    if frame.event_id_cnty.duplicated().any() or frame.duplicate_group_id.duplicated().any():
        raise ValueError("Training data contains duplicate IDs or groups")
    if frame.final_label.nunique() != 21 or frame.final_label.isna().any():
        raise ValueError("Training data must contain exactly 21 non-empty labels")
    heldout = pd.read_csv(heldout_path)
    if set(frame.event_id_cnty) & set(heldout.event_id_cnty):
        raise ValueError("Training data contains held-out event IDs")
    if set(frame.duplicate_group_id) & set(heldout.duplicate_group_id):
        raise ValueError("Training data contains held-out duplicate groups")


def build_dev_selector(gold_path, manifest_path, tokenizer, class_names, device, max_len, batch_size):
    gold = pd.read_csv(gold_path, keep_default_na=False)
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    expected = set(manifest.event_id_cnty)
    if set(gold.event_id_cnty) != expected or gold.event_id_cnty.duplicated().any():
        raise ValueError("Development gold must exactly and uniquely cover the frozen dev manifest")
    hashes = manifest[["event_id_cnty", "note_hash"]].merge(
        gold[["event_id_cnty", "note_hash"]], on="event_id_cnty", suffixes=("_manifest", "_gold"), validate="one_to_one"
    )
    if not hashes.note_hash_manifest.eq(hashes.note_hash_gold).all():
        raise ValueError("Development gold note hashes do not match the frozen manifest")
    if not gold.primary_label.isin(class_names).all():
        raise ValueError("Development gold contains labels absent from the model mapping")
    ordered = manifest[["event_id_cnty"]].merge(gold, on="event_id_cnty", validate="one_to_one")
    texts = [build_input(row) for _, row in ordered.iterrows()]

    def select(model):
        predictions = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                encoded = tokenizer(
                    texts[start:start + batch_size], truncation=True, max_length=max_len,
                    padding=True, return_tensors="pt",
                ).to(device)
                predictions.extend(model(**encoded).logits.argmax(dim=-1).cpu().tolist())
        names = [class_names[index] for index in predictions]
        strict = [prediction == primary for prediction, primary in zip(names, ordered.primary_label)]
        accepted_rows = [
            accepted(prediction, primary, tuple(item for item in alternatives.split("|") if item))
            for prediction, primary, alternatives in zip(names, ordered.primary_label, ordered.alternative_labels)
        ]
        return {"n": len(ordered), "strict": float(np.mean(strict)), "accepted": float(np.mean(accepted_rows))}

    return select


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/release/labeled_balanced_21.csv")
    parser.add_argument("--model", default="models/attempt7-t5-leave-keywords-in-epoch-2/hf_transformer_model")
    parser.add_argument("--output", default="models/release21-t5")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--heldout-manifest", default="data/run/heldout_manifest.csv")
    parser.add_argument("--dev-gold", default="data/run/dev_gold.csv")
    parser.add_argument("--dev-manifest", default="data/run/dev_manifest.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--training-mode", choices=["full", "reviewed", "mixed"], default="full")
    parser.add_argument("--reviewed-fraction", type=float, default=0.5)
    args = parser.parse_args()
    data_path = Path(args.data)
    validate_training_data(data_path, Path(args.heldout_manifest))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    full = pd.read_csv(data_path, low_memory=False)
    reviewed_mask = full.review_status.isin(["independent_review", "human_review", "adjudicated"])
    if args.training_mode == "reviewed":
        prepared = full[reviewed_mask].copy()
    elif args.training_mode == "mixed":
        if not 0 < args.reviewed_fraction < 1:
            raise ValueError("--reviewed-fraction must be between 0 and 1")
        n_reviewed = round(len(full) * args.reviewed_fraction)
        n_weak = len(full) - n_reviewed
        prepared = pd.concat([
            full[reviewed_mask].sample(n=n_reviewed, replace=True, random_state=args.seed),
            full[~reviewed_mask].sample(n=n_weak, replace=n_weak > (~reviewed_mask).sum(), random_state=args.seed),
        ]).sample(frac=1, random_state=args.seed)
    else:
        prepared = full
    if args.training_mode != "full":
        data_path = Path("data/run") / f"_{args.training_mode}_training.csv"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        prepared.to_csv(data_path, index=False)
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
    splits = load_data(
        data_path, train_split=0.8, val_split=0.1, random_state=args.seed,
        class_names=CLASS_NAMES,
    )
    if len(splits.class_names) != 21 or "other" not in splits.class_names:
        raise ValueError(f"Expected 21 labels including other, got {splits.class_names}")
    device, device_name = get_device()
    tokenizer = load_tokenizer(args.model)
    loaders = build_dataloaders(splits, tokenizer, args.max_len, args.batch_size)
    dev_selector = build_dev_selector(
        Path(args.dev_gold), Path(args.dev_manifest), tokenizer, splits.class_names,
        device, args.max_len, args.batch_size,
    )
    model, train_losses, val_losses, training_time = train_model(
        args.model, splits.class_names, loaders[0], loaders[1], device,
        learning_rate=args.learning_rate, epochs=args.epochs, output_dir=args.output, tokenizer=tokenizer,
        freeze_encoder=args.freeze_encoder,
        early_stopping_patience=args.patience, min_delta=args.min_delta,
        selection_fn=dev_selector,
    )
    save_model(model, tokenizer, args.output)
    metrics = {
        "model": args.model, "output": args.output, "device": device_name,
        "epochs": args.epochs, "batch_size": args.batch_size, "max_len": args.max_len,
        "freeze_encoder": args.freeze_encoder,
        "seed": args.seed, "learning_rate": args.learning_rate,
        "training_mode": args.training_mode, "reviewed_fraction": args.reviewed_fraction,
        "source_reviewed_rows": int(reviewed_mask.sum()),
        "early_stopping_patience": args.patience, "min_delta": args.min_delta,
        "epochs_completed": len(train_losses),
        "best_epoch": model.best_epoch, "dev_selection_history": model.selection_history,
        "training_rows": len(splits.train), "validation_rows": len(splits.val), "test_rows": len(splits.test),
        "class_names": splits.class_names, "train_losses": train_losses, "val_losses": val_losses,
        "training_time_seconds": training_time,
        "weak_validation_diagnostic": evaluate_model(model, loaders[1], device),
        "weak_test_diagnostic": evaluate_model(model, loaders[2], device),
    }
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "release21_metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n")
    print(json.dumps(metrics, indent=2, default=float))


if __name__ == "__main__":
    main()
