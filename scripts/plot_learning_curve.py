#!/usr/bin/env python3
"""Create the development learning-curve JSON and plot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from protest_classifier.evaluation.learning_curve import extract_learning_curve

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/new_classifier_dev.json")
    parser.add_argument("--output-json", type=Path, default=ROOT / "reports/new_classifier_learning_curve.json")
    parser.add_argument("--output-png", type=Path, default=ROOT / "reports/new_classifier_learning_curve.png")
    args = parser.parse_args()
    curve = extract_learning_curve(json.loads(args.report.read_text()))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"curve": curve}, indent=2) + "\n")
    sizes = [point["size"] for point in curve]
    accuracy = [point["mean_strict_accuracy"] for point in curve]
    macro_f1 = [point["mean_macro_f1"] for point in curve]
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(sizes, accuracy, "o-", label="strict accuracy")
    axis.plot(sizes, macro_f1, "s--", label="macro-F1")
    axis.set(xlabel="Training annotations", ylabel="Development score", title="Learning curve")
    axis.set_xscale("log", base=2)
    axis.set_xticks(sizes, [str(size) for size in sizes])
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output_png, dpi=150)
    print(f"wrote {args.output_json} and {args.output_png}")


if __name__ == "__main__":
    main()
