"""Select the year-balanced, duplicate-safe experimental training set."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REVIEWED_STATUSES = {"independent_review", "human_review", "adjudicated"}


def sample_spread(frame: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Sample without replacement while round-robining countries."""
    if n <= 0:
        return frame.head(0)
    shuffled = frame.iloc[rng.permutation(len(frame))].copy()
    shuffled["_country_rank"] = shuffled.groupby("country", dropna=False).cumcount()
    return shuffled.sort_values(["_country_rank", "country"], kind="stable").head(n).drop(columns="_country_rank")


def quotas_for_class(frame: pd.DataFrame, budget: int) -> dict[int, int]:
    available = frame.groupby("year").size().sort_index()
    budget = min(int(budget), int(available.sum()))
    if not budget:
        return {int(y): 0 for y in available.index}
    years = list(available.index)
    requested = {int(y): min(int(available.loc[y]), budget // len(years)) for y in years}
    remaining = budget - sum(requested.values())
    # Redistribute the coverage shortfall in deterministic year order.
    while remaining:
        room = {y: int(available.loc[y]) - requested[y] for y in years if int(available.loc[y]) > requested[y]}
        if not room:
            break
        for year in sorted(room):
            if remaining == 0:
                break
            requested[year] += 1
            remaining -= 1
    # Half the budget is coverage; the remaining half follows residual availability.
    coverage = {y: min(requested[y], budget // 2 // len(years)) for y in years}
    left = budget - sum(coverage.values())
    residual = {y: int(available.loc[y]) - coverage[y] for y in years}
    while left:
        eligible = [y for y in years if residual[y] > 0]
        if not eligible:
            break
        total = sum(residual[y] for y in eligible)
        raw = {y: left * residual[y] / total for y in eligible}
        add = {y: min(residual[y], int(np.floor(raw[y]))) for y in eligible}
        if sum(add.values()) == 0:
            for y in sorted(eligible, key=lambda x: (-raw[x], x))[:left]:
                add[y] = 1
        for y, n in add.items():
            coverage[y] += n
            residual[y] -= n
        left = budget - sum(coverage.values())
    return coverage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=Path("data/release/training_candidates.csv"))
    parser.add_argument("--heldout", type=Path, default=Path("data/run/heldout_manifest.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/release"))
    parser.add_argument("--budget", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(args.candidates)
    heldout = pd.read_csv(args.heldout, usecols=["event_id_cnty", "duplicate_group_id"])
    held_ids = set(heldout.event_id_cnty)
    held_groups = set(heldout.duplicate_group_id)
    candidates["event_date"] = pd.to_datetime(candidates["event_date"], errors="coerce")
    candidates["year"] = candidates.event_date.dt.year.astype("Int64")
    eligible = candidates[
        candidates.final_label.notna()
        & ~candidates.event_id_cnty.isin(held_ids)
        & ~candidates.duplicate_group_id.isin(held_groups)
    ].drop_duplicates("duplicate_group_id").copy()
    rng = np.random.default_rng(args.seed)
    selected = []
    coverage = []
    for label in sorted(eligible.final_label.unique()):
        pool = eligible[eligible.final_label.eq(label)]
        quotas = quotas_for_class(pool, args.budget)
        for year, requested in quotas.items():
            cell = pool[pool.year.eq(year)]
            # Reviewed rows are guaranteed a place before weak-only rows are sampled.
            reviewed = cell[cell.review_status.isin(REVIEWED_STATUSES)]
            weak = cell[~cell.review_status.isin(REVIEWED_STATUSES)]
            n = min(requested, len(reviewed) + len(weak))
            if n:
                take_reviewed = min(n, len(reviewed))
                chosen_reviewed = sample_spread(reviewed, take_reviewed, rng)
                take_weak = n - take_reviewed
                chosen_weak = sample_spread(weak, take_weak, rng)
                chosen = pd.concat([chosen_reviewed, chosen_weak])
                selected.append(chosen)
            coverage.append({"class": label, "year": year, "available": len(cell), "requested": requested, "selected": n, "human_reviewed": int(cell.review_status.isin(REVIEWED_STATUSES).sum())})
    if not selected:
        raise ValueError("No eligible labeled candidates")
    release = pd.concat(selected, ignore_index=True)
    if release.duplicate_group_id.duplicated().any():
        raise AssertionError("Selection contains duplicate groups")
    release.to_csv(args.out / "labeled_balanced_21.csv", index=False)
    coverage_df = pd.DataFrame(coverage)
    coverage_df["quota_gap"] = coverage_df.requested - coverage_df.selected
    coverage_df.to_csv(args.out / "year_coverage.csv", index=False)
    release[["event_id_cnty", "duplicate_group_id", "final_label", "year", "country"]].to_csv(
        args.out / "selected_training_ids.csv", index=False
    )
    country = release.groupby(["final_label", "year", "country"], dropna=False).size().rename("selected").reset_index()
    totals = country.groupby(["final_label", "year"]).selected.transform("sum")
    country["country_share"] = country.selected / totals
    country.to_csv(args.out / "country_coverage.csv", index=False)

    print(f"selected {len(release):,} rows; {int(release.review_status.isin(REVIEWED_STATUSES).sum()):,} independently reviewed")
    print(coverage_df[coverage_df.quota_gap.ne(0)].to_string(index=False))


if __name__ == "__main__":
    main()
