"""Train the standalone 21-class ModernBERT protest classifier.

One command trains every requested seed on one nested training release and saves
the best development checkpoint for each seed.  The model reads the original
``notes`` string only; there are no keyword features, no metadata and no model
input beyond the note text.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from classifier_v2.common import (
    CLASSIFIER_DATA_DIR,
    CLASS_NAMES,
    MODELS_DIR,
    normalize_notes,
    reject_empty_notes,
    label_maps,
)

MODEL_NAME = "answerdotai/ModernBERT-base"


class NotesDataset(Dataset):
    """Holds unpadded token encodings; the collator pads dynamically per batch."""

    def __init__(self, encodings: dict, labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = int(self.labels[index])
        return item


@dataclass
class Release:
    name: str
    train: pd.DataFrame
    dev: pd.DataFrame


def load_release(train_csv: Path, dev_csv: Path) -> Release:
    train = pd.read_csv(train_csv, low_memory=False)
    dev = pd.read_csv(dev_csv, low_memory=False)
    for frame, name in ((train, "training"), (dev, "development")):
        if "notes" not in frame or "primary_label" not in frame:
            raise ValueError(f"{name} file must contain notes and primary_label")
        frame["notes"] = frame["notes"].map(normalize_notes)
        reject_empty_notes(frame, context=name)
        invalid = sorted(set(frame["primary_label"].astype(str)) - set(CLASS_NAMES))
        if invalid:
            raise ValueError(f"{name} contains labels outside the taxonomy: {invalid}")
    return Release(train_csv.stem, train, dev)


def encode(tokenizer, texts: list[str], max_len: int) -> dict:
    # Measure truncation on the un-truncated tokenization.
    full = tokenizer(texts, add_special_tokens=True)
    truncated = sum(len(ids) > max_len for ids in full["input_ids"])
    encodings = tokenizer(texts, truncation=True, max_length=max_len, add_special_tokens=True)
    encodings["_truncated"] = truncated
    return encodings


def build_compute_metrics():
    from sklearn.metrics import accuracy_score, f1_score

    all_labels = list(range(len(CLASS_NAMES)))

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            # Average over the full 21-class taxonomy, matching evaluation, so
            # development selection cannot ignore classes absent from the dev set.
            "macro_f1": float(
                f1_score(labels, predictions, labels=all_labels, average="macro", zero_division=0)
            ),
        }

    return compute_metrics


def _make_trainer(model, args, train_dataset, eval_dataset, collator, tokenizer, patience):
    kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=build_compute_metrics(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )
    try:
        return Trainer(processing_class=tokenizer, **kwargs)
    except TypeError:
        return Trainer(tokenizer=tokenizer, **kwargs)


def train_one_seed(
    release: Release,
    *,
    model_name: str = MODEL_NAME,
    output_dir: Path,
    seed: int = 42,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    epochs: int = 10,
    per_device_batch_size: int = 8,
    grad_accum: int = 4,
    max_len: int = 512,
    patience: int = 2,
) -> dict:
    set_seed(seed)
    id2label, label2id = label_maps()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_enc = encode(tokenizer, release.train["notes"].tolist(), max_len)
    dev_enc = encode(tokenizer, release.dev["notes"].tolist(), max_len)
    truncated_train = train_enc.pop("_truncated")
    truncated_dev = dev_enc.pop("_truncated")

    train_labels = [label2id[label] for label in release.train["primary_label"]]
    dev_labels = [label2id[label] for label in release.dev["primary_label"]]
    train_dataset = NotesDataset(train_enc, train_labels)
    dev_dataset = NotesDataset(dev_enc, dev_labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(CLASS_NAMES),
        id2label=id2label,
        label2id=label2id,
    )
    model.config.problem_type = "single_label_classification"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bf16 = bool(torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)())

    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size * 2,
        gradient_accumulation_steps=grad_accum,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_strategy="epoch",
        seed=seed,
        data_seed=seed,
        bf16=bf16,
        report_to="none",
        save_only_model=True,
    )

    trainer = _make_trainer(
        model, args, train_dataset, dev_dataset,
        DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
        tokenizer, patience,
    )
    train_result = trainer.train()

    best_dir = output_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    log_history = [entry for entry in trainer.state.log_history if "eval_macro_f1" in entry]
    best_entry = max(log_history, key=lambda entry: entry["eval_macro_f1"]) if log_history else {}
    metrics = {
        "model": model_name,
        "run": release.name,
        "seed": seed,
        "output_dir": str(best_dir),
        "train_rows": len(release.train),
        "dev_rows": len(release.dev),
        "effective_batch_size": per_device_batch_size * grad_accum,
        "per_device_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "max_len": max_len,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "max_epochs": epochs,
        "early_stopping_patience": patience,
        "warmup_ratio": 0.1,
        "bf16": bf16,
        "epochs_completed": trainer.state.epoch,
        "best_epoch": best_entry.get("epoch"),
        "best_dev_macro_f1": best_entry.get("eval_macro_f1"),
        "best_dev_accuracy": best_entry.get("eval_accuracy"),
        "truncated_train_notes": int(truncated_train),
        "truncated_dev_notes": int(truncated_dev),
        "train_loss": train_result.training_loss,
        "dev_history": log_history,
        "class_distribution": release.train["primary_label"].value_counts().to_dict(),
    }
    (Path(output_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n")
    print(json.dumps({k: metrics[k] for k in (
        "run", "seed", "train_rows", "dev_rows", "best_epoch",
        "best_dev_macro_f1", "best_dev_accuracy", "truncated_train_notes",
    )}, indent=2, default=float))
    return metrics


def resolve_release_paths(release: int | None, train_csv: Path | None, dev_csv: Path) -> tuple[Path, Path]:
    if train_csv is None:
        if release is None:
            raise SystemExit("Provide --release (e.g. 800) or --train-csv")
        train_csv = CLASSIFIER_DATA_DIR / f"train_{release:04d}.csv"
    if not Path(train_csv).exists():
        raise SystemExit(f"Training release not found: {train_csv}")
    if not Path(dev_csv).exists():
        raise SystemExit(f"Development set not found: {dev_csv}")
    return Path(train_csv), Path(dev_csv)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=int, default=None, choices=[800, 1500, 3000, 6000])
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--dev-csv", type=Path, default=CLASSIFIER_DATA_DIR / "dev_497.csv")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Per-device batch size; effective size = batch-size * grad-accum")
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--patience", type=int, default=2)
    args = parser.parse_args()

    train_csv, dev_csv = resolve_release_paths(args.release, args.train_csv, args.dev_csv)
    release = load_release(train_csv, dev_csv)
    run_name = train_csv.stem.replace("train_", "run-")
    print(f"Training {run_name}: {len(release.train)} train / {len(release.dev)} dev rows")

    for seed in args.seeds:
        output_dir = args.output_dir or (MODELS_DIR / run_name / f"seed-{seed}")
        train_one_seed(
            release,
            model_name=args.model,
            output_dir=output_dir,
            seed=seed,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            per_device_batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            max_len=args.max_len,
            patience=args.patience,
        )


if __name__ == "__main__":
    main()
