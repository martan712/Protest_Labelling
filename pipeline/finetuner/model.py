"""Model loading, training, evaluation and saving."""

import os
import time

import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

MODEL_SUBDIR = "hf_transformer_model"


def get_device():
    """Return (device, human-readable name), preferring GPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps"), "MPS (Apple Silicon GPU)"
    if torch.cuda.is_available():
        return torch.device("cuda"), f"CUDA ({torch.cuda.get_device_name(0)})"
    return torch.device("cpu"), "CPU"


def load_tokenizer(model_name):
    try:
        return AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        raise ValueError(f"Could not load tokenizer for model '{model_name}'. Error: {e}")


def _load_model(model_name, class_names):
    # Bake real class names into the config so the saved model is self-describing
    # at inference (no need to reconstruct the mapping from the training CSV).
    id2label = {i: name for i, name in enumerate(class_names)}
    label2id = {name: i for i, name in id2label.items()}
    try:
        return AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(class_names), id2label=id2label, label2id=label2id
        )
    except Exception as e:
        raise ValueError(
            f"Could not load model '{model_name}' for sequence classification. Error: {e}"
        )


def _autocast(device):
    # bf16 roughly halves GPU compute/memory with no procedure change and needs
    # no GradScaler. No-op off CUDA.
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def train_model(model_name, class_names, train_loader, val_loader, device,
                learning_rate, epochs, output_dir=None, tokenizer=None):
    model = _load_model(model_name, class_names).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    train_losses, val_losses = [], []
    start = time.time()

    print("\nStarting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            optimizer.zero_grad(set_to_none=True)
            batch = {k: v.to(device) for k, v in batch.items()}
            with _autocast(device):
                loss = model(**batch).loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_losses.append(total_loss / len(train_loader))
        val_losses.append(evaluate_model(model, val_loader, device)["avg_loss"])
        print(f"Epoch {epoch}: train_loss={train_losses[-1]:.4f}, val_loss={val_losses[-1]:.4f}")

        # Checkpoint every epoch so a crash doesn't lose progress; same path each
        # time, so it doubles as the final model.
        if output_dir is not None and tokenizer is not None:
            save_model(model, tokenizer, output_dir)

    return model, train_losses, val_losses, time.time() - start


def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            with _autocast(device):
                outputs = model(**batch)
            total_loss += outputs.loss.item()
            predictions.extend(outputs.logits.argmax(dim=1).cpu().numpy())
            true_labels.extend(batch["labels"].cpu().numpy())

    labels = sorted(set(true_labels) | set(predictions))
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average="weighted", zero_division=0, labels=labels
    )
    return {
        "avg_loss": total_loss / len(data_loader),
        "accuracy": accuracy_score(true_labels, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def save_model(model, tokenizer, output_dir):
    path = os.path.join(output_dir, MODEL_SUBDIR)
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Model and tokenizer saved to {path}")
