"""Sample unused ACLED events for additional semantic annotations.

Events are drawn uniformly at random from ``data/filtered_events.csv`` after
excluding every development ID, every test ID, every existing training ID, every
already-queued ID and every duplicate note hash.  The output is a blind
annotation queue: it carries the original notes and empty label columns only —
no old class, no keyword, no ``clean_notes`` and no model prediction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from classifier_v2.common import (
    ANNOTATION_SOURCES,
    CLASSIFIER_DATA_DIR,
    DATA_DIR,
    DEV_GOLD,
    DEV_REMOVE_IDS,
    REPO_ROOT,
    TEST_MANIFEST_840,
    _resolve,
    load_annotations,
    normalize_notes,
    sha256_text,
)

FILTERED_EVENTS = DATA_DIR / "filtered_events.csv"
DEFAULT_OUT_DIR = DATA_DIR / "review" / "sampled_v1"

QUEUE_COLUMNS = (
    "annotation_order", "event_id_cnty", "event_date", "year", "country",
    "notes", "note_hash", "primary_label", "alternative_labels", "other_reason",
    "evidence", "reviewer", "annotation_status", "taxonomy_version",
)


def _ids_and_hashes(frame: pd.DataFrame) -> tuple[set[str], set[str]]:
    ids = set(frame.get("event_id_cnty", pd.Series(dtype=str)).astype(str))
    hashes = set(frame.get("note_hash", pd.Series(dtype=str)).astype(str))
    return ids, hashes


def _existing_queue_paths(sources=ANNOTATION_SOURCES) -> list[Path]:
    paths: list[Path] = []
    for _, pattern in sources:
        resolved = _resolve(pattern)
        paths.extend(sorted(resolved.parent.glob(resolved.name)))
    # Legacy queues that were never labelled.
    for pattern in (
        "data/review/expanded_v1/annotation_queue.csv",
        "data/review/unknown_starter_v1/annotation_queue.csv",
        "data/review/test_gold_v1/annotation_queue.csv",
        "data/review/test_v2/chunks/chunk_*.csv",
    ):
        resolved = _resolve(pattern)
        paths.extend(sorted(resolved.parent.glob(resolved.name)))
    return paths


def excluded_events(filtered_path: Path = FILTERED_EVENTS) -> tuple[set[str], set[str]]:
    """All IDs / note hashes that must never be sampled."""
    exclude_ids: set[str] = set()
    exclude_hashes: set[str] = set()

    for path in (DEV_GOLD, TEST_MANIFEST_840):
        if Path(path).exists():
            ids, hashes = _ids_and_hashes(pd.read_csv(path, low_memory=False))
            exclude_ids |= ids
            exclude_hashes |= hashes
    # The three dev IDs that also appear in test.
    exclude_ids |= set(DEV_REMOVE_IDS)

    annotations = load_annotations()
    ids, hashes = _ids_and_hashes(annotations)
    exclude_ids |= ids
    exclude_hashes |= hashes

    for path in _existing_queue_paths():
        if not Path(path).exists():
            continue
        frame = pd.read_csv(path, low_memory=False)
        frame = frame.copy()
        if "note_hash" not in frame and "notes" in frame:
            frame["note_hash"] = frame["notes"].map(lambda x: sha256_text(normalize_notes(x)))
        ids, hashes = _ids_and_hashes(frame)
        exclude_ids |= ids
        exclude_hashes |= hashes
    return exclude_ids, exclude_hashes


def queued_rows(sources=ANNOTATION_SOURCES) -> int:
    """Number of annotation-source rows that are queued but not yet labelled.

    Only the chunk files that :func:`prepare_training_data` will actually load
    count, so legacy ``annotation_queue.csv`` copies (which duplicate chunk rows
    or hold test candidates) cannot inflate the number.
    """
    seen: set[str] = set()
    for _, pattern in sources:
        resolved = _resolve(pattern)
        for path in sorted(resolved.parent.glob(resolved.name)):
            frame = pd.read_csv(path, low_memory=False)
            if "primary_label" in frame:
                frame = frame[frame["primary_label"].isna()]
            if "event_id_cnty" in frame:
                seen |= set(frame["event_id_cnty"].astype(str))
    return len(seen)


def sample_unused(
    n: int,
    *,
    seed: int = 42,
    filtered_path: Path = FILTERED_EVENTS,
    exclude_ids: set[str] | None = None,
    exclude_hashes: set[str] | None = None,
) -> pd.DataFrame:
    """Return ``n`` uniformly sampled, unused events (one row per note hash)."""
    if exclude_ids is None or exclude_hashes is None:
        exclude_ids, exclude_hashes = excluded_events(filtered_path)

    frame = pd.read_csv(
        filtered_path,
        usecols=["event_id_cnty", "event_date", "year", "country", "notes"],
        low_memory=False,
    )
    frame = frame[~frame["event_id_cnty"].astype(str).isin(exclude_ids)].copy()
    frame["notes"] = frame["notes"].map(normalize_notes)
    frame = frame[frame["notes"].str.len() > 0].copy()
    frame["note_hash"] = frame["notes"].map(sha256_text)
    frame = frame[~frame["note_hash"].isin(exclude_hashes)]
    frame = frame.drop_duplicates(subset="note_hash", keep="first")

    if n > len(frame):
        raise ValueError(f"Requested {n} events but only {len(frame)} unused events remain")
    return frame.sample(n=n, random_state=seed).sort_values("event_id_cnty").reset_index(drop=True)


def write_queue(
    sampled: pd.DataFrame,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    chunk_size: int = 100,
    seed: int = 42,
    target: int | None = None,
) -> list[Path]:
    out_dir = Path(out_dir)
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    queue = sampled.copy()
    queue.insert(0, "annotation_order", range(1, len(queue) + 1))
    for column in ("primary_label", "alternative_labels", "other_reason", "evidence", "reviewer"):
        queue[column] = ""
    queue["annotation_status"] = "pending"
    queue["taxonomy_version"] = "21-class-v1"
    queue = queue[list(QUEUE_COLUMNS)]

    queue.to_csv(out_dir / "annotation_queue.csv", index=False)
    selection = sampled[["event_id_cnty", "note_hash"]].copy()
    selection["duplicate_group_id"] = "note-" + selection["note_hash"].str[:16]
    selection["selection_reason"] = "random_unused"
    selection = selection[["event_id_cnty", "duplicate_group_id", "selection_reason"]]
    selection.to_csv(out_dir / "selection_manifest.csv", index=False)

    paths: list[Path] = []
    for index, start in enumerate(range(0, len(queue), chunk_size), start=1):
        path = chunks_dir / f"chunk_{index:02d}.csv"
        queue.iloc[start:start + chunk_size].to_csv(path, index=False)
        paths.append(path)

    metadata = {
        "queue_version": "sampled-v1",
        "taxonomy_version": "21-class-v1",
        "seed": seed,
        "target": target,
        "rows": len(queue),
        "chunk_size": chunk_size,
        "chunk_count": len(paths),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return paths


def needed_for(target: int) -> int:
    """How many *new* events must be sampled to reach ``target`` training rows."""
    available = len(load_annotations())
    need = target - available - queued_rows()
    return max(0, need)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=1500,
                        help="Cumulative training size to prepare a queue for.")
    parser.add_argument("--n", type=int, default=None,
                        help="Override: sample exactly this many new events.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--filtered", type=Path, default=FILTERED_EVENTS)
    args = parser.parse_args()

    n = args.n if args.n is not None else needed_for(args.target)
    print(f"Labelled: {len(load_annotations())}, already queued: {queued_rows()}, sampling: {n}")
    if n == 0:
        print("Nothing to sample; the queue already reaches the target.")
        return
    sampled = sample_unused(n, seed=args.seed, filtered_path=args.filtered)
    paths = write_queue(sampled, args.out_dir, chunk_size=args.chunk_size, seed=args.seed, target=args.target)
    print(f"Wrote {len(sampled)} events to {args.out_dir} ({len(paths)} chunks)")


if __name__ == "__main__":
    main()
