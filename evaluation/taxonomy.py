"""Shared taxonomy and scoring definitions for the 21-class label release."""

from __future__ import annotations

TAXONOMY_VERSION = "21-class-v1"
CLASS_NAMES = [
    "animal welfare",
    "blm",
    "climate",
    "culture",
    "discrimination",
    "education",
    "environment",
    "farmers",
    "health care",
    "housing",
    "immigration",
    "labor rights",
    "lgbtq",
    "other",
    "palestine-israel conflict",
    "pandemic",
    "policies & politics",
    "public services",
    "ukraine-russia war",
    "unjust law enforcement",
    "women rights",
]

ACCEPTED_PAIRS = {frozenset(p) for p in [
    ("climate", "environment"),
    ("unjust law enforcement", "blm"),
    ("public services", "health care"),
    ("public services", "education"),
    ("public services", "housing"),
    ("discrimination", "women rights"),
    ("discrimination", "lgbtq"),
    ("discrimination", "blm"),
]}

OTHER_REASONS = {"outside_taxonomy", "insufficient_information"}


def accepted(prediction: str, primary: str, alternatives: tuple[str, ...] | list[str] = ()) -> bool:
    """Return whether a prediction matches a gold label or one direct accepted pair."""
    return any(
        prediction == gold or frozenset((prediction, gold)) in ACCEPTED_PAIRS
        for gold in {primary, *alternatives}
    )


def validate_label(label: str, other_reason: str | None = None) -> None:
    if label not in CLASS_NAMES:
        raise ValueError(f"Unknown label {label!r}; expected one of {CLASS_NAMES}")
    if label == "other" and other_reason not in OTHER_REASONS:
        raise ValueError("other requires outside_taxonomy or insufficient_information")
    if label != "other" and other_reason not in (None, ""):
        raise ValueError("other_reason is only valid for the other class")
