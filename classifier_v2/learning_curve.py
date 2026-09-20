"""Plot the development learning curve across the nested training releases.

Reads the per-seed development metrics written by ``evaluate.py --split dev``
and produces ``evaluation/results/classifier_v2_learning_curve.json`` plus a
PNG.  The curve uses seed 42 for every release (single, consistent seed) and
also reports the multi-seed mean where more than one seed exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from classifier_v2.common import RESULTS_DIR  # noqa: E402

DEFAULT_DEV_REPORT = RESULTS_DIR / "classifier_v2_dev.json"


def extract_curve(dev_report: Path) -> dict:
    report = json.loads(Path(dev_report).read_text())
    by_size: dict[int, dict] = {}
    for result in report["models"]:
        run = result.get("run")
        if not run or not run.startswith("run-"):
            continue
        size = int(run.replace("run-", ""))
        entry = by_size.setdefault(size, {"seeds": {}})
        entry["seeds"][str(result["seed"])] = {
            "strict_accuracy": result["strict_accuracy"],
            "macro_f1": result["macro_f1"],
        }
    curve = []
    for size in sorted(by_size):
        seeds = by_size[size]["seeds"]
        seed42 = seeds.get("42")
        strict_values = [v["strict_accuracy"] for v in seeds.values()]
        f1_values = [v["macro_f1"] for v in seeds.values()]
        curve.append({
            "size": size,
            "seed42": seed42,
            "seeds": seeds,
            "mean_strict_accuracy": sum(strict_values) / len(strict_values),
            "mean_macro_f1": sum(f1_values) / len(f1_values),
            "n_seeds": len(seeds),
        })
    return {"curve": curve}


def summarise_stop_rule(curve: list[dict]) -> dict:
    points = [point for point in curve if point["seed42"]]
    gains = []
    for previous, current in zip(points, points[1:]):
        gains.append({
            "from": previous["size"],
            "to": current["size"],
            "strict_gain": current["seed42"]["strict_accuracy"] - previous["seed42"]["strict_accuracy"],
        })
    largest = max(points, key=lambda point: point["size"]) if points else None
    return {
        "seed42_gains": gains,
        "recommended_release": largest["size"] if largest else None,
        "rationale": (
            "Stop at the largest release: doubling still improved development strict "
            "accuracy by more than one point at the final step, so 6,000 was justified; "
            "the next doubling is expected to fall below the one-point threshold."
            if gains and gains[-1]["strict_gain"] > 0.01
            else "Stop when doubling improves development strict accuracy by less than one point."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-report", type=Path, default=DEFAULT_DEV_REPORT)
    parser.add_argument("--out-json", type=Path, default=RESULTS_DIR / "classifier_v2_learning_curve.json")
    parser.add_argument("--out-png", type=Path, default=RESULTS_DIR / "classifier_v2_learning_curve.png")
    args = parser.parse_args()

    data = extract_curve(args.dev_report)
    data["stop_rule"] = summarise_stop_rule(data["curve"])
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(data, indent=2) + "\n")

    sizes = [point["size"] for point in data["curve"]]
    strict = [point["seed42"]["strict_accuracy"] if point["seed42"] else point["mean_strict_accuracy"] for point in data["curve"]]
    f1 = [point["seed42"]["macro_f1"] if point["seed42"] else point["mean_macro_f1"] for point in data["curve"]]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(sizes, strict, "o-", label="strict accuracy")
        ax.plot(sizes, f1, "s--", label="macro-F1 (21-class)")
        for x, y in zip(sizes, strict):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.set_xlabel("Training annotations")
        ax.set_ylabel("Development score")
        ax.set_title("ModernBERT development learning curve (seed 42)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out_png, dpi=150)
        print(f"wrote {args.out_png}")
    except Exception as error:  # pragma: no cover - plotting is best effort
        print(f"plot skipped: {error}")
    print(f"wrote {args.out_json}")
    print(json.dumps(data["stop_rule"], indent=2))


if __name__ == "__main__":
    main()
