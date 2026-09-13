"""Plot the completed ablation records; requires matplotlib only for this report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "baseline": "Original Full",
    "frequency": "Eigenvalue labels",
    "capacity_control": "MLP with zero labels",
    "full_kernel": "Complete spectrum kernel",
    "diagonal": "Kernel diagonal",
    "combined": "All three changes",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads((args.study / "results.json").read_text())
    plan = json.loads((args.study / "plan.json").read_text())
    seeds = plan["configuration"]["seeds"]
    names = list(plan["arms"])
    values = np.array(
        [
            [
                next(r["best_val_mae"] for r in rows if r["arm"] == name and r["seed"] == seed)
                for seed in seeds
            ]
            for name in names
        ]
    )
    mean, std = values.mean(1), values.std(1, ddof=1)
    baseline = values[names.index("baseline")]
    diff = values - baseline[None, :]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.edgecolor": "#bac1c7",
            "text.color": "#24313b",
            "axes.labelcolor": "#24313b",
            "xtick.color": "#43505a",
            "ytick.color": "#24313b",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(11.5, 4.8), sharey=True, gridspec_kw={"width_ratios": [1.1, 1]}
    )
    y = np.arange(len(names))
    axes[0].errorbar(
        mean,
        y,
        xerr=std,
        fmt="s",
        color="#176876",
        capsize=4,
        markersize=5,
        linewidth=1.7,
        label="Mean +/- sample SD",
    )
    for index, seed in enumerate(seeds):
        offset = (index - (len(seeds) - 1) / 2) * 0.10
        axes[0].scatter(
            values[:, index],
            y + offset,
            s=15,
            facecolors="white",
            edgecolors="#76828b",
            zorder=3,
            label="Individual seeds" if index == 0 else None,
        )
        axes[1].scatter(
            diff[:, index],
            y + offset,
            s=23,
            marker=["o", "^", "x"][index % 3],
            color="#526a76",
            label=f"Seed {seed}",
        )
    axes[0].axvline(baseline.mean(), color="#9ea7ae", linestyle="--", linewidth=1)
    axes[1].axvline(0, color="#9ea7ae", linestyle="--", linewidth=1)
    axes[1].scatter(diff.mean(1), y, marker="|", s=220, linewidths=2, color="#176876", label="Mean")
    axes[0].set_yticks(y, [LABELS[name] for name in names])
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Model configuration")
    axes[0].set_xlabel("Best validation MAE (target units)")
    axes[1].set_xlabel("Paired MAE difference vs original Full")
    axes[0].set_title("Validation error", loc="left", fontweight="bold", y=1.14)
    axes[1].set_title(
        "Negative differences favor the change", loc="left", fontweight="bold", y=1.14
    )
    for ax in axes:
        ax.grid(axis="x", color="#ebedf0", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    axes[0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, fontsize=8)
    axes[1].legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=4, fontsize=8)
    fig.suptitle(
        f"Spectral information ablation | ZINC | {plan['configuration']['epochs']} epochs",
        x=0.025,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.025,
        0.015,
        "Source: frozen local study, 2026-09-05; audited 2026-09-06. "
        "10,000 train / 1,000 validation graphs; "
        "three seeds; validation-selected checkpoints. Test split not loaded.",
        fontsize=8,
        color="#69757e",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.93), w_pad=2.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    fig.savefig(args.output.with_suffix(".svg"))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
