"""Tokenization and PyTorch dataset adapters."""

from __future__ import annotations

from torch.utils.data import Dataset


class NotesDataset(Dataset):
    def __init__(self, encodings: dict, labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = int(self.labels[index])
        return item


def tokenize(tokenizer, texts: list[str], *, max_length: int) -> tuple[dict, int]:
    full = tokenizer(texts, add_special_tokens=True)
    truncated = sum(len(token_ids) > max_length for token_ids in full["input_ids"])
    encoded = tokenizer(texts, truncation=True, max_length=max_length, add_special_tokens=True)
    return encoded, truncated
