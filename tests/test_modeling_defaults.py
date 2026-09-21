from __future__ import annotations

import unittest

import torch

from protest_classifier.modeling.inference import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_READ_CHUNK_SIZE,
    inference_dtype,
)
from protest_classifier.modeling.datasets import tokenize
from protest_classifier.modeling.training import TrainingConfig


class RecordingTokenizer:
    def __init__(self) -> None:
        self.call_sizes: list[int] = []

    def __call__(self, texts, *, return_length=False, truncation=False, max_length=None, **_):
        self.call_sizes.append(len(texts))
        lengths = [len(text.split()) + 2 for text in texts]
        if return_length:
            return {"length": lengths}
        if truncation:
            lengths = [min(length, max_length) for length in lengths]
        return {
            "input_ids": [list(range(length)) for length in lengths],
            "attention_mask": [[1] * length for length in lengths],
        }


class ModelingDefaultsTests(unittest.TestCase):
    def test_training_tokenization_is_chunked_and_counts_truncation(self) -> None:
        tokenizer = RecordingTokenizer()
        encoded, truncated = tokenize(
            tokenizer,
            ["one two", "one two three four", "one"],
            max_length=5,
            chunk_size=2,
        )
        self.assertEqual(tokenizer.call_sizes, [2, 2, 1, 1])
        self.assertEqual(truncated, 1)
        self.assertEqual([len(ids) for ids in encoded["input_ids"]], [4, 5, 3])

    def test_training_defaults_keep_effective_batch_and_reduce_padding(self) -> None:
        config = TrainingConfig()
        self.assertEqual(config.batch_size * config.gradient_accumulation, 32)
        self.assertEqual(config.max_length, 256)
        self.assertTrue(config.group_by_length)
        self.assertTrue(config.gradient_checkpointing)

    def test_inference_defaults_are_bounded_and_batched(self) -> None:
        self.assertEqual(DEFAULT_BATCH_SIZE, 16)
        self.assertEqual(DEFAULT_MAX_LENGTH, 256)
        self.assertEqual(DEFAULT_READ_CHUNK_SIZE, 1024)
        self.assertEqual(
            inference_dtype(torch.device("cpu"), use_bfloat16=True),
            torch.float32,
        )


if __name__ == "__main__":
    unittest.main()
