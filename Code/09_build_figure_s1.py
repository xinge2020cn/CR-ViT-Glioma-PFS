"""Build Supplementary Figure S1 from the patient-level segmentation metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260902
COHORTS = ["Training cohort", "Temporal validation cohort", "Spatial validation cohort"]
SHORT = ["Training", "Temporal\nvalidation", "Spatial\nvalidation"]
COLORS = ["#274D6B", "#51708A", "#7890A3"]


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8A949C")
    ax.spines["bottom"].set_color("#8A949C")
    ax.tick_params(axis="both", colors="#28343D", labelsize=8.5, width=0.6)
    ax.grid(axis="y", color="#E8EBED", linewidth=0.65, zorder=0)


def add_distribution(
    ax: plt.Axes,
    data: pd.DataFrame,
    column: str,
    summary_text: list[str],
    ylim: tuple[float, float],
    log_scale: bool = False,
) -> None:
    rng = np.random.default_rng(SEED + (1 if column == "dice_similarity" else 2))
    values = [data.loc[data["cohort"] == cohort, column].to_numpy() for cohort in COHORTS]
    bp = ax.boxplot(
        values,
        positions=np.arange(1, 4),
        widths=0.50,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        medianprops={"color": "white", "linewidth": 1.3},
        boxprops={"linewidth": 0.8, "edgecolor": "#203846"},
        whiskerprops={"linewidth": 0.8, "color": "#52636F"},
        capprops={"linewidth": 0.8, "color": "#52636F"},
        zorder=3,
    )
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.93)

    for pos, cohort, color in zip(range(1, 4), COHORTS, COLORS):
        cohort_values = data.loc[data["cohort"] == cohort, column].to_numpy()
        x = pos + rng.uniform(-0.25, 0.25, size=len(cohort_values))
        ax.scatter(x, cohort_values, s=5.2, alpha=0.16, color=color, linewidths=0, zorder=2)

    if log_scale:
        ax.set_yscale("log")
        ax.set_yticks([1, 2, 4, 8, 16])
        ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.get_yaxis().set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_ylim(*ylim)
    ax.set_xticks([1, 2, 3], SHORT)
    style_axis(ax)

    for pos, text in zip(range(1, 4), summary_text):
        ax.text(
            pos,
            1.025,
            text,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7.6,
            color="#263640",
            clip_on=False,
        )


def main() -> None:
    data = pd.read_csv(ROOT / "Data" / "segmentation_metrics_patient_level.csv")
    summary = pd.read_csv(ROOT / "Results" / "segmentation_performance.csv")
    summary = summary.set_index("cohort").loc[COHORTS].reset_index()

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.weight": "bold",
            "font.size": 9,
            "axes.labelcolor": "#1F2D36",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    dice_text = [f"{r.dice_mean:.3f} ± {r.dice_sd:.3f}" for r in summary.itertuples()]
    hd_text = [
        f"{r.hd95_median_mm:.2f} ({r.hd95_q1_mm:.2f}–{r.hd95_q3_mm:.2f})"
        for r in summary.itertuples()
    ]
    component_dir = ROOT / "Figures" / "_ppt_ready_components" / "S1_segmentation"
    component_dir.mkdir(parents=True, exist_ok=True)
    specifications = [
        ("A_dice.png", "dice_similarity", dice_text, (0.56, 1.005), False,
         "Dice similarity coefficient"),
        ("B_hd95.png", "hd95_mm", hd_text, (0.70, 22.0), True,
         "HD95, mm (log scale)"),
    ]
    for filename, column, labels, ylim, log_scale, ylabel in specifications:
        fig, ax = plt.subplots(figsize=(3.55, 2.70), constrained_layout=False)
        fig.patch.set_facecolor("white")
        add_distribution(ax, data, column, labels, ylim, log_scale)
        ax.set_ylabel(ylabel, fontsize=9.1)
        fig.subplots_adjust(left=0.18, right=0.985, top=0.79, bottom=0.20)
        fig.savefig(component_dir / filename, dpi=600, facecolor="white")
        plt.close(fig)
    print(f"Wrote independent Figure S1 panels to {component_dir}")


if __name__ == "__main__":
    main()
