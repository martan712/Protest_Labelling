"""Data loading, tokenization and batching for fine-tuning."""

import os
from dataclasses import dataclass

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .triggers import strip_triggers

TEXT_COLUMN = "clean_notes"
CLASS_COLUMN = "class"
CLASS_NAMES = [
    "animal welfare", "blm", "climate", "culture", "discrimination", "education",
    "environment", "farmers", "health care", "housing", "immigration", "labor rights",
    "lgbtq", "other", "palestine-israel conflict", "pandemic", "policies & politics",
    "public services", "ukraine-russia war", "unjust law enforcement", "women rights",
]
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


def build_input(row, include_year=True, include_country=True):
    """Build the one structured input used by training and inference."""
    note = row.get("notes", row.get(TEXT_COLUMN, ""))
    parts = []
    if include_year and pd.notna(row.get("year")):
        parts.append(f"Event year: {int(row['year'])}")
    if include_country and pd.notna(row.get("country")):
        parts.append(f"Country: {row['country']}")
    parts.append(f"Note: {str(note)}")
    return "\n".join(parts)


def load_data(file_path, train_split=0.8, val_split=0.1, random_state=42,
              strip_trigger_words=False, include_year=True, include_country=True) -> Splits:
    """Load the labeled CSV and split it into train/val/test.

    Labels are 0-based indices into a sorted list of class names; inference must
    rebuild the same sorted order (see config.id2label on the saved model).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    df = pd.read_csv(file_path)
    if CLASS_COLUMN not in df and "final_label" in df:
        df = df.rename(columns={"final_label": CLASS_COLUMN})
    if "notes" not in df:
        df["notes"] = df[TEXT_COLUMN]
    df[TEXT_COLUMN] = df.apply(lambda row: build_input(row, include_year, include_country), axis=1)
    df = df[~df[CLASS_COLUMN].isin(DROP_CLASSES)]

    df = df.dropna(subset=[TEXT_COLUMN]).copy()
    df = df[df[TEXT_COLUMN].str.strip() != ""]

    # Optionally strip the labeler's trigger keywords so the model can't keyword-match.
    # Off by default: the keywords carry real signal. Drops rows left empty afterwards.
    if strip_trigger_words:
        before = len(df)
        df[TEXT_COLUMN] = df[TEXT_COLUMN].map(strip_triggers)
        df = df[df[TEXT_COLUMN].str.strip() != ""]
        print(f"Stripped trigger words; dropped {before - len(df)} now-empty rows ({len(df)} remain)")

    class_names = [name for name in CLASS_NAMES if name in set(df[CLASS_COLUMN])]
    class_to_label = {name: i for i, name in enumerate(class_names)}
    df["label"] = df[CLASS_COLUMN].map(class_to_label).astype(int)

    total = len(df)
    train_size = int(train_split * total)
    val_size = int(val_split * total)

    # Keep duplicate note groups together whenever provenance is available.
    if "duplicate_group_id" in df:
        groups = df[["duplicate_group_id"]].drop_duplicates().sample(frac=1, random_state=random_state)
        n_train_groups = max(1, round(len(groups) * train_split))
        train_groups = set(groups.head(n_train_groups).duplicate_group_id)
        train_df = df[df.duplicate_group_id.isin(train_groups)]
        if len(train_df) > train_size:
            train_df = train_df.sample(n=train_size, random_state=random_state)
    else:
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
