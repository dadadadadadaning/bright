#!/usr/bin/env python3
"""Plot reported historical numbers; this script does not evaluate a model.

No confidence intervals or missing checkpoints are inferred. CSV provenance
must remain attached when figures are exported.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path(__file__).with_name("reported_results.csv"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("figures"))
    args = parser.parse_args()
    with args.csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row["split"] != "test878" for row in rows):
        raise ValueError("This figure accepts Test878 rows only; never mix Val and Test.")
    if len({row["seed"] for row in rows}) != 1:
        raise ValueError("This is a single-trajectory figure, not a multi-seed summary.")
    data = {}
    for row in rows:
        key = (int(row["step"]), row["arm"])
        if key in data:
            raise ValueError(f"Duplicate row: {key}")
        data[key] = float(row["miou4_pct"])
    steps = [500, 1000, 2500]
    required = {(step, arm) for step in steps for arm in ("grid", "e", "f", "b", "d", "r", "l")}
    if set(data) != required:
        raise ValueError("Expected exactly the three reported complete checkpoints and seven arms.")

    # This script can run on read-only-home containers without changing HOME.
    with tempfile.TemporaryDirectory(prefix="bright_plot_cache_") as cache:
        os.environ.setdefault("MPLCONFIGDIR", cache)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                             "svg.fonttype": "none", "axes.spines.top": False,
                             "axes.spines.right": False})
        fig, axes = plt.subplots(1, 2, figsize=(12.3, 5.1), gridspec_kw={"width_ratios": [1.05, 1.0]})
        colors = {"e": "#3B6EA8", "b": "#C47F22", "d": "#198572"}
        labels = {"e": "E: local, single slot", "b": "B: local, dual slot", "d": "D: historical, dual slot"}
        for arm in ("e", "b", "d"):
            gains = [data[step, arm] - data[step, "grid"] for step in steps]
            # Only complete reported endpoints are drawn. Lines connect observations;
            # they are not claims about unmeasured intermediate checkpoints.
            axes[0].plot(steps, gains, marker="o", markersize=5, linewidth=1.7,
                         color=colors[arm], label=labels[arm])
        axes[0].set(title="A. Overall gain at reported checkpoints", xlabel="Training step",
                    ylabel="Four-class mIoU gain over Grid (percentage points)",
                    xticks=steps, ylim=(0, 0.61))
        axes[0].grid(axis="y", alpha=0.22)
        axes[0].legend(loc="lower right", frameon=False, fontsize=9)

        contrasts = [("B − E", "b", "e"), ("D − F", "d", "f"),
                     ("D − B", "d", "b"), ("D − R", "d", "r"),
                     ("D − L", "d", "l")]
        differences = [data[2500, left] - data[2500, right] for _, left, right in contrasts]
        axes[1].barh(range(len(contrasts)), differences,
                     color=["#198572" if value >= 0 else "#BA4450" for value in differences],
                     height=0.57)
        axes[1].set(yticks=range(len(contrasts)), yticklabels=[item[0] for item in contrasts],
                    title="B. Component contrasts at step 2500",
                    xlabel="Four-class mIoU difference (percentage points)", xlim=(-0.004, 0.024))
        axes[1].invert_yaxis()
        axes[1].axvline(0, color="#555555", linewidth=0.8)
        axes[1].grid(axis="x", alpha=0.22)
        for y, value in enumerate(differences):
            axes[1].text(value + (0.00045 if value >= 0 else -0.00045), y,
                         f"{value:+.4f}", va="center", ha="left" if value >= 0 else "right", fontsize=9)
        fig.suptitle("BRIGHT: reported Test878 results, seed 3407", fontsize=14, y=0.98)
        fig.text(0.5, 0.025,
                 "User-reported complete results; raw reports not revalidated here. "
                 "One trajectory; no uncertainty estimates.\n"
                 "Test1500/2000 omitted because complete evidence was not supplied. "
                 "Panel B uses a much narrower scale.",
                 ha="center", va="bottom", fontsize=8, color="#555555")
        fig.tight_layout(rect=(0, 0.12, 1, 0.93))
        args.output.mkdir(parents=True, exist_ok=True)
        for suffix in ("svg", "png"):
            path = args.output / f"reported_test_gains.{suffix}"
            fig.savefig(path, dpi=180, facecolor="white")
            print(path)
        plt.close(fig)


if __name__ == "__main__":
    main()
