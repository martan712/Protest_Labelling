"""Canonical label mapping and checkpoint validation."""

from __future__ import annotations

from pathlib import Path

from ..taxonomy import CLASS_NAMES


def label_maps() -> tuple[dict[int, str], dict[str, int]]:
    id_to_label = dict(enumerate(CLASS_NAMES))
    return id_to_label, {label: index for index, label in id_to_label.items()}


def validate_checkpoint_labels(model) -> dict[int, str]:
    expected, _ = label_maps()
    actual = {int(index): label for index, label in model.config.id2label.items()}
    if actual != expected:
        raise ValueError(f"Checkpoint label mapping differs from taxonomy: {actual}")
    return expected


def find_checkpoints(models_dir: Path, runs: list[str]) -> list[Path]:
    checkpoints: list[Path] = []
    for run in runs:
        run_dir = models_dir / run
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        for seed_dir in sorted(run_dir.glob("seed-*")):
            best = seed_dir / "best"
            checkpoints.append(best if best.exists() else seed_dir)
    return checkpoints
