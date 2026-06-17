"""Data loading, tokenization and batching for fine-tuning."""

import os
from dataclasses import dataclass

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .triggers import strip_triggers

TEXT_COLUMN = "clean_notes"
CLASS_COLUMN = "class"
# Placeholders that are not real categories; 'unknown' especially must be
# dropped so the model never learns to emit it (it exists to be replaced).
DROP_CLASSES = ("NoN", "unknown")


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    class_names: list[str]  # 0-based: index i == model output i

    @property
    def num_labels(self) -> int:
        return len(self.class_names)

    def class_counts(self, split: pd.DataFrame) -> dict[str, int]:
        names = pd.Series(self.class_names)
        return split["label"].map(names).value_counts().to_dict()


class TextDataset(Dataset):
    """Holds unpadded token lists; collate_fn pads per batch."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = int(self.labels[idx])
        return item


def load_data(file_path, train_split=0.8, val_split=0.1, random_state=42) -> Splits:
    """Load the labeled CSV and split it into train/val/test.

    Labels are 0-based indices into a sorted list of class names; inference must
    rebuild the same sorted order (see config.id2label on the saved model).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    df = pd.read_csv(file_path)[[CLASS_COLUMN, TEXT_COLUMN]]
    df = df[~df[CLASS_COLUMN].isin(DROP_CLASSES)]

    # strip the labeler's trigger keywords from the training text so the model
    # can't keyword-match. Drop rows left empty (they had no signal beyond the keyword).
    df = df.dropna(subset=[TEXT_COLUMN]).copy()
    before = len(df)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].map(strip_triggers)
    df = df[df[TEXT_COLUMN].str.strip() != ""]
    print(f"Stripped trigger words; dropped {before - len(df)} now-empty rows ({len(df)} remain)")

    class_names = sorted(df[CLASS_COLUMN].unique())
    class_to_label = {name: i for i, name in enumerate(class_names)}
    df["label"] = df[CLASS_COLUMN].map(class_to_label).astype(int)

    total = len(df)
    train_size = int(train_split * total)
    val_size = int(val_split * total)

    train_df = df.sample(n=train_size, random_state=random_state)
    remaining = df.drop(train_df.index)
    val_df = remaining.sample(n=val_size, random_state=random_state)
    test_df = remaining.drop(val_df.index)

    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Dataset loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    return Splits(train_df, val_df, test_df, class_names)


def _make_collate_fn(tokenizer):
    """Pad each batch to its own longest sequence (dynamic padding)."""

    def collate(features):
        labels = torch.tensor([int(f.pop("labels")) for f in features], dtype=torch.long)
        batch = tokenizer.pad(features, padding="longest", return_tensors="pt")
        batch["labels"] = labels
        return batch

    return collate


def _to_loader(df, tokenizer, max_len, batch_size, shuffle):
    # No padding/tensors here: the collate_fn pads dynamically per batch.
    encodings = tokenizer(df[TEXT_COLUMN].tolist(), truncation=True, max_length=max_len)
    dataset = TextDataset(encodings, df["label"].tolist())
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_make_collate_fn(tokenizer)
    )


def build_dataloaders(splits: Splits, tokenizer, max_len=128, batch_size=16):
    return (
        _to_loader(splits.train, tokenizer, max_len, batch_size, shuffle=True),
        _to_loader(splits.val, tokenizer, max_len, batch_size, shuffle=False),
        _to_loader(splits.test, tokenizer, max_len, batch_size, shuffle=False),
    )
