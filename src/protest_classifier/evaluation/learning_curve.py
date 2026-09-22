"""Aggregate development results across nested training sizes."""

from __future__ import annotations


def extract_learning_curve(report: dict) -> list[dict]:
    by_size: dict[int, dict[str, dict]] = {}
    for result in report["models"]:
        run = result.get("run", "")
        if not run.startswith("run-"):
            continue
        size = int(run.removeprefix("run-"))
        by_size.setdefault(size, {})[str(result.get("seed"))] = {
            "strict_accuracy": result["strict_accuracy"],
            "macro_f1": result["macro_f1"],
        }
    curve = []
    for size, seeds in sorted(by_size.items()):
        curve.append({
            "size": size,
            "seeds": seeds,
            "seed42": seeds.get("42"),
            "mean_strict_accuracy": sum(row["strict_accuracy"] for row in seeds.values()) / len(seeds),
            "mean_macro_f1": sum(row["macro_f1"] for row in seeds.values()) / len(seeds),
            "n_seeds": len(seeds),
        })
    return curve
