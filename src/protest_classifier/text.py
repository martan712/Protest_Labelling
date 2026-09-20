"""Canonical preprocessing for model input text."""

from __future__ import annotations

import hashlib
import re

import pandas as pd

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_WHITESPACE = re.compile(r"\s+")


def normalize_notes(value: object) -> str:
    """Remove control characters and collapse whitespace without changing wording."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).replace("\u00a0", " ")
    return _WHITESPACE.sub(" ", _CONTROL_CHARS.sub(" ", text)).strip()


def note_hash(value: object) -> str:
    """Hash normalized note text using the project-wide convention."""
    return hashlib.sha256(normalize_notes(value).encode("utf-8")).hexdigest()
