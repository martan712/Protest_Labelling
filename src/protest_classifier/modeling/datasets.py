"""Tokenization and PyTorch dataset adapters."""

from __future__ import annotations

from torch.utils.data import Dataset

TOKENIZE_CHUNK_SIZE = 512


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


def tokenize(
    tokenizer,
    texts: list[str],
    *,
    max_length: int,
    chunk_size: int = TOKENIZE_CHUNK_SIZE,
) -> tuple[dict, int]:
    """Tokenize without retaining a second, untruncated copy of the corpus."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    encoded: dict[str, list] = {}
    truncated = 0
    for start in range(0, len(texts), chunk_size):
        text_chunk = texts[start:start + chunk_size]
        lengths = tokenizer(
            text_chunk,
            add_special_tokens=True,
            return_length=True,
        )["length"]
        truncated += sum(length > max_length for length in lengths)
        encoded_chunk = tokenizer(
            text_chunk,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        for key, values in encoded_chunk.items():
            encoded.setdefault(key, []).extend(values)
    return encoded, truncated
