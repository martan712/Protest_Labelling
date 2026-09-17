"""Prediction decoders for strict argmax and optional direct-pair mass scoring."""

from __future__ import annotations

import torch

from evaluation.taxonomy import ACCEPTED_PAIRS, accepted


def accepted_mass_decode(logits: torch.Tensor, class_names: list[str], gold_labels: list[set[str]]) -> list[str]:
    """Choose labels by probability mass over direct accepted pairs for dev experiments."""
    probabilities = logits.softmax(dim=-1)
    outputs = []
    for row, golds in zip(probabilities, gold_labels):
        scores = []
        for candidate in class_names:
            compatible = [name for name in golds if name in class_names and (
                name == candidate or frozenset((name, candidate)) in ACCEPTED_PAIRS
            )]
            scores.append(row[[class_names.index(name) for name in compatible]].sum() if compatible else row[class_names.index(candidate)])
        outputs.append(class_names[int(torch.tensor(scores).argmax())])
    return outputs


def accepted_for_row(prediction: str, primary: str, alternatives: tuple[str, ...] = ()) -> bool:
    return accepted(prediction, primary, alternatives)
