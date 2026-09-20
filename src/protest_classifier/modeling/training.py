"""ModernBERT fine-tuning without project paths or command-line behavior."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from ..data.training_sets import TrainingRelease
from ..taxonomy import CLASS_NAMES
from .checkpoints import label_maps
from .datasets import NotesDataset, tokenize


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str = "answerdotai/ModernBERT-base"
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 10
    batch_size: int = 8
    gradient_accumulation: int = 4
    max_length: int = 512
    patience: int = 2


def compute_metrics(eval_prediction) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    label_ids = list(range(len(CLASS_NAMES)))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, labels=label_ids, average="macro", zero_division=0)
        ),
    }


def _trainer(model, arguments, train_data, dev_data, collator, tokenizer, patience):
    kwargs = dict(
        model=model,
        args=arguments,
        train_dataset=train_data,
        eval_dataset=dev_data,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )
    try:
        return Trainer(processing_class=tokenizer, **kwargs)
    except TypeError:
        return Trainer(tokenizer=tokenizer, **kwargs)


def train(
    release: TrainingRelease,
    output_dir: Path,
    *,
    seed: int,
    config: TrainingConfig = TrainingConfig(),
) -> dict:
    """Train one seed, save its best checkpoint, and return run metrics."""
    set_seed(seed)
    id_to_label, label_to_id = label_maps()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    train_encodings, truncated_train = tokenize(
        tokenizer, release.train.notes.tolist(), max_length=config.max_length
    )
    dev_encodings, truncated_dev = tokenize(
        tokenizer, release.dev.notes.tolist(), max_length=config.max_length
    )
    train_data = NotesDataset(
        train_encodings, [label_to_id[label] for label in release.train.primary_label]
    )
    dev_data = NotesDataset(
        dev_encodings, [label_to_id[label] for label in release.dev.primary_label]
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(CLASS_NAMES),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    model.config.problem_type = "single_label_classification"
    output_dir.mkdir(parents=True, exist_ok=True)
    use_bfloat16 = bool(
        torch.cuda.is_available()
        and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    )
    arguments = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,
        gradient_accumulation_steps=config.gradient_accumulation,
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
        bf16=use_bfloat16,
        report_to="none",
        save_only_model=True,
    )
    trainer = _trainer(
        model,
        arguments,
        train_data,
        dev_data,
        DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
        tokenizer,
        config.patience,
    )
    result = trainer.train()
    best_dir = output_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    history = [entry for entry in trainer.state.log_history if "eval_macro_f1" in entry]
    best = max(history, key=lambda entry: entry["eval_macro_f1"]) if history else {}
    metrics = {
        "model": config.model_name,
        "run": release.name,
        "seed": seed,
        "output_dir": str(best_dir),
        "train_rows": len(release.train),
        "dev_rows": len(release.dev),
        "config": asdict(config),
        "effective_batch_size": config.batch_size * config.gradient_accumulation,
        "bf16": use_bfloat16,
        "epochs_completed": trainer.state.epoch,
        "best_epoch": best.get("epoch"),
        "best_dev_macro_f1": best.get("eval_macro_f1"),
        "best_dev_accuracy": best.get("eval_accuracy"),
        "truncated_train_notes": truncated_train,
        "truncated_dev_notes": truncated_dev,
        "train_loss": result.training_loss,
        "dev_history": history,
        "class_distribution": release.train.primary_label.value_counts().to_dict(),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n")
    return metrics
