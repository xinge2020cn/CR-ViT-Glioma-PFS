from pathlib import Path
from math import exp, log, sqrt
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "Results"
DATA = ROOT / "Data"
FIGURES = ROOT / "Figures"
OUT = FIGURES / "_ppt_ready_components"

for sub in [
    "Figure3",
    "S2_calibration",
    "S3_dca",
    "S4_brier",
    "S5_pairwise",
    "S6_vit_km",
]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)

for font_path in [
    r"C:\Windows\Fonts\times.ttf",
    r"C:\Windows\Fonts\timesbd.ttf",
    r"C:\Windows\Fonts\timesi.ttf",
    r"C:\Windows\Fonts\timesbi.ttf",
]:
    if Path(font_path).exists():
        font_manager.fontManager.addfont(font_path)

mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.linewidth": 0.9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

NAVY = "#1F3B57"
DARK = "#202A33"
GRID = "#E2E7EB"
MODEL_COLORS = {
    "Clinicoradiologic": "#7A7A7A",
    "3D-CNN": "#56A0C8",
    "3D-ViT": "#D8892B",
    "CR-ViT": "#274C77",
}
MODELS = ["Clinicoradiologic", "3D-CNN", "3D-ViT", "CR-ViT"]
SUBTYPES = ["IDHmut-intact", "IDHmut-codel", "GBM"]
SUBTYPE_LABELS = {
    "IDHmut-intact": "IDHmut-intact",
    "IDHmut-codel": "IDHmut-codel",
    "GBM": "GBM",
}
SUBTYPE_COLORS = {
    "IDHmut-intact": "#5B8DB8",
    "IDHmut-codel": "#5AAE8A",
    "GBM": "#C65A3A",
}
MAIN_COHORTS = ["Training cohort", "Temporal validation cohort", "Spatial validation cohort"]
ALL_COHORTS = MAIN_COHORTS + ["GBM multi-omics subset"]
COHORT_SHORT = {
    "Training cohort": "Training cohort",
    "Temporal validation cohort": "Temporal validation cohort",
    "Spatial validation cohort": "Spatial validation cohort",
    "GBM multi-omics subset": "GBM multi-omics subset",
}
COHORT_SLUG = {
    "Training cohort": "training",
    "Temporal validation cohort": "temporal",
    "Spatial validation cohort": "spatial",
    "GBM multi-omics subset": "multiomics",
}
SUBTYPE_SLUG = {
    "IDHmut-intact": "idhmut_intact",
    "IDHmut-codel": "idhmut_codel",
    "GBM": "gbm",
}

perf = pd.read_csv(RESULTS / "clinicoradiologic_crvit_performance.csv")
td = pd.read_csv(RESULTS / "clinicoradiologic_crvit_time_dependent_metrics.csv")
cal = pd.read_csv(RESULTS / "clinicoradiologic_crvit_calibration.csv")
dca = pd.read_csv(RESULTS / "clinicoradiologic_crvit_decision_curve.csv")
brier = pd.read_csv(RESULTS / "clinicoradiologic_crvit_diagnostic_brier.csv")
comp = pd.read_csv(RESULTS / "clinicoradiologic_crvit_paired_comparisons.csv")
pred = pd.read_csv(RESULTS / "clinicoradiologic_crvit_patient_predictions.csv")
km_stats = pd.read_csv(RESULTS / "clinicoradiologic_crvit_km_statistics.csv")
vit_km_stats = pd.read_csv(RESULTS / "vit_km_statistics.csv")
bridge = pd.read_csv(RESULTS / "vit_clinicoradiologic_bridge_associations.csv")
bridge_r2 = pd.read_csv(RESULTS / "vit_clinicoradiologic_bridge_crossvalidated_r2.csv")
table3_cox = pd.read_csv(RESULTS / "Table3_validation_multivariable_cox.csv")
multi_ids = set(pd.read_csv(DATA / "center1_gbm_multiomics_subset_patient_level.csv")["patient_id"])


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def save_component(fig, folder, name, tight=False):
    path = OUT / folder / name
    save_kwargs = {"dpi": 450, "facecolor": "white"}
    if tight:
        save_kwargs.update({"bbox_inches": "tight", "pad_inches": 0.02})
    fig.savefig(path, **save_kwargs)
    plt.close(fig)
    return path


def style_axis(ax, grid="both"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid, color=GRID, linewidth=0.65, zorder=0)
    ax.set_axisbelow(True)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def legend_box(ax, handles, loc="upper right", fontsize=6.2, ncol=1, title=None):
    leg = ax.legend(
        handles=handles,
        loc=loc,
        ncol=ncol,
        fontsize=fontsize,
        frameon=True,
        fancybox=True,
        framealpha=0.78,
        facecolor="white",
        edgecolor="#B7C0C8",
        title=title,
        borderpad=0.32,
        handlelength=1.35,
        handletextpad=0.35,
        columnspacing=0.65,
        labelspacing=0.20,
    )
    if title:
        leg.get_title().set_fontweight("bold")
        leg.get_title().set_fontsize(fontsize)
    try:
        leg.set_alignment("left")
    except AttributeError:
        pass
    for text in leg.get_texts():
        text.set_fontweight("bold")
    return leg


model_handles = [Line2D([0], [0], color=MODEL_COLORS[m], marker="o", lw=1.6,
                        markersize=4, label=m) for m in MODELS]
bar_handles = [Patch(facecolor=MODEL_COLORS[m], edgecolor="none", label=m) for m in MODELS]


def draw_cindex_by_subtype(ax, cohort):
    bar_width = 0.18
    xbase = np.arange(len(SUBTYPES))
    for mi, model in enumerate(MODELS):
        xpos = xbase + (mi - 1.5) * bar_width
        rows = [
            perf[(perf.cohort == cohort) & (perf.subtype == st) & (perf.model == model)].iloc[0]
            for st in SUBTYPES
        ]
        est = np.array([r.cindex for r in rows], dtype=float)
        lo = np.array([r.cindex_low for r in rows], dtype=float)
        hi = np.array([r.cindex_high for r in rows], dtype=float)
        bars = ax.bar(xpos, est - 0.50, bottom=0.50, width=bar_width * 0.90,
                      color=MODEL_COLORS[model], edgecolor="white", linewidth=0.35, zorder=2)
        ax.errorbar(xpos, est, yerr=[est - lo, hi - est], fmt="none",
                    ecolor=DARK, elinewidth=0.75, capsize=1.8, zorder=3)
        for bar, value, upper in zip(bars, est, hi):
            ax.text(bar.get_x() + bar.get_width() / 2, upper + 0.011, f"{value:.2f}",
                    ha="center", va="bottom", fontsize=6.4, clip_on=False)
    ax.set_xticks(xbase, [SUBTYPE_LABELS[s] for s in SUBTYPES])
    ax.set_ylim(0.50, 0.90)
    ax.set_yticks(np.arange(0.5, 0.91, 0.1))
    ax.set_ylabel("Harrell C-index")
    style_axis(ax, "y")
    legend_box(ax, bar_handles, loc="upper right", fontsize=5.6, ncol=2)


def draw_cindex_gbm_multiomics(ax):
    z = perf[(perf.cohort == "GBM multi-omics subset") & (perf.subtype == "GBM")]
    z = z.set_index("model").loc[MODELS]
    x = np.arange(len(MODELS))
    est = z.cindex.to_numpy(float)
    lo = z.cindex_low.to_numpy(float)
    hi = z.cindex_high.to_numpy(float)
    bars = ax.bar(x, est - 0.50, bottom=0.50, width=0.62,
                  color=[MODEL_COLORS[m] for m in MODELS],
                  edgecolor="white", linewidth=0.4, zorder=2)
    ax.errorbar(x, est, yerr=[est - lo, hi - est], fmt="none",
                ecolor=DARK, elinewidth=0.8, capsize=2.0, zorder=3)
    for bar, value, upper in zip(bars, est, hi):
        ax.text(bar.get_x() + bar.get_width() / 2, upper + 0.012, f"{value:.2f}",
                ha="center", va="bottom", fontsize=7.0, clip_on=False)
    ax.set_xticks(x, MODELS, rotation=25, ha="right")
    ax.set_ylim(0.50, 0.86)
    ax.set_yticks(np.arange(0.5, 0.81, 0.1))
    ax.set_ylabel("Harrell C-index")
    style_axis(ax, "y")


def draw_auc(ax, cohort, subtype, legend_loc):
    for model in MODELS:
        z = td[(td.cohort == cohort) & (td.subtype == subtype) & (td.model == model)].sort_values("time_months")
        ax.plot(z.time_months, z.auc, color=MODEL_COLORS[model], lw=1.65,
                marker="o", markersize=3.1, clip_on=True)
    ax.set_xlim(5.8, 36.2)
    ax.set_ylim(0.49, 1.02)
    ax.set_xticks([6, 12, 18, 24, 30, 36])
    ax.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax.set_xlabel("Time since surgery (months)")
    ax.set_ylabel("Time-dependent AUC")
    style_axis(ax, "both")
    legend_box(ax, model_handles, loc=legend_loc, fontsize=6.5, ncol=2)


def km_curve(frame, group, group_col="crvit_risk_group", xmax=36.0):
    d = frame[frame[group_col] == group].copy().sort_values("pfs_time_months")
    times = d.pfs_time_months.to_numpy(float)
    events = d.event.to_numpy(int)
    event_times = np.unique(times[(events == 1) & (times <= xmax)])
    x = [0.0]
    svals = [1.0]
    lower = [1.0]
    upper = [1.0]
    surv = 1.0
    greenwood = 0.0
    for tm in event_times:
        n_risk = np.sum(times >= tm)
        n_event = np.sum((times == tm) & (events == 1))
        if n_risk <= 0:
            continue
        surv *= (1.0 - n_event / n_risk)
        if n_risk > n_event:
            greenwood += n_event / (n_risk * (n_risk - n_event))
        if 0 < surv < 1 and greenwood > 0:
            se_loglog = sqrt(greenwood) / abs(log(surv))
            lo = exp(-exp(log(-log(surv)) + 1.96 * se_loglog))
            hi = exp(-exp(log(-log(surv)) - 1.96 * se_loglog))
        else:
            lo = hi = surv
        x.append(float(tm))
        svals.append(float(surv))
        lower.append(lo)
        upper.append(hi)
    if x[-1] < xmax:
        x.append(xmax)
        svals.append(svals[-1])
        lower.append(lower[-1])
        upper.append(upper[-1])
    cens = times[(events == 0) & (times <= xmax)]
    cens_surv = []
    xa = np.asarray(x)
    sa = np.asarray(svals)
    for tm in cens:
        idx = np.searchsorted(xa, tm, side="right") - 1
        cens_surv.append(sa[max(idx, 0)])
    return np.asarray(x), np.asarray(svals), np.asarray(lower), np.asarray(upper), cens, np.asarray(cens_surv)


def p_fmt(p):
    return r"$P$ < 0.001" if p < 0.001 else rf"$P$ = {p:.3f}"


def draw_km(main_ax, risk_ax, frame, stat_row, legend_loc="upper right",
            group_col="crvit_risk_group"):
    risk_colors = {"Low risk": "#3F7FB5", "High risk": "#C65A3A"}
    for group in ["Low risk", "High risk"]:
        x, s, lo, hi, ct, cs = km_curve(frame, group, group_col=group_col)
        main_ax.step(x, s, where="post", color=risk_colors[group], lw=1.55, clip_on=True)
        main_ax.fill_between(x, lo, hi, step="post", color=risk_colors[group], alpha=0.10, clip_on=True)
        if len(ct):
            take = np.linspace(0, len(ct) - 1, min(len(ct), 16)).astype(int)
            main_ax.plot(ct[take], cs[take], linestyle="none", marker="|", markersize=3.0,
                         color=risk_colors[group], alpha=0.55, clip_on=True)
    main_ax.set_xlim(0, 36)
    main_ax.set_ylim(0, 1.00)
    main_ax.set_xticks([0, 6, 12, 18, 24, 30, 36])
    main_ax.tick_params(labelbottom=False, pad=1.5)
    main_ax.set_ylabel("PFS probability", labelpad=2)
    style_axis(main_ax, "both")
    hr_title = (
        f"HR {stat_row.high_vs_low_hr:.2f} "
        f"({stat_row.ci_low:.2f}\u2013{stat_row.ci_high:.2f})\n"
        f"Log-rank {p_fmt(stat_row.logrank_p_value)}"
    )
    handles = [Line2D([0], [0], color=risk_colors[g], lw=1.8, label=g)
               for g in ["Low risk", "High risk"]]
    legend_box(main_ax, handles, loc=legend_loc, fontsize=10.0, title=hr_title)

    times = [0, 6, 12, 18, 24, 30, 36]
    risk_ax.set_xlim(0, 36)
    risk_ax.set_ylim(0, 1)
    risk_ax.set_xticks(times)
    risk_ax.set_xlabel("Time since surgery (months)", labelpad=2)
    risk_ax.set_yticks([0.54, 0.20], ["", ""])
    risk_ax.tick_params(axis="x", length=2, pad=1.5, labelsize=5.8)
    risk_ax.tick_params(axis="y", length=0, pad=0, labelsize=5.8)
    risk_ax.text(-0.12, 0.90, "Number at risk", transform=risk_ax.transAxes,
                 ha="left", va="center", fontsize=6.0, clip_on=False)
    for yi, group in zip([0.54, 0.20], ["Low risk", "High risk"]):
        d = frame[frame[group_col] == group]
        risk_ax.text(-0.055, yi, group, transform=risk_ax.transAxes,
                     ha="right", va="center", fontsize=5.7,
                     color=risk_colors[group], clip_on=False)
        for tm in times:
            risk_ax.text(tm, yi, str(int(np.sum(d.pfs_time_months >= tm))),
                         ha="center", va="center", fontsize=5.6, color=risk_colors[group],
                         transform=risk_ax.get_xaxis_transform(), clip_on=False)
    for side in ["top", "right", "left"]:
        risk_ax.spines[side].set_visible(False)
    risk_ax.spines["bottom"].set_color("#AAB4BC")
    risk_ax.grid(False)


def draw_km_component(frame, stat_row, folder, name, group_col, legend_loc):
    fig = plt.figure(figsize=(4.35, 2.45))
    inner = fig.add_gridspec(2, 1, left=0.18, right=0.985, bottom=0.15, top=0.97,
                             height_ratios=[4.0, 1.25], hspace=0.03)
    ax = fig.add_subplot(inner[0])
    rt = fig.add_subplot(inner[1])
    draw_km(ax, rt, frame, stat_row, legend_loc=legend_loc, group_col=group_col)
    save_component(fig, folder, name)


def draw_calibration(ax, cohort, subtype):
    z0 = cal[(cal.cohort == cohort) & (cal.subtype == subtype)]
    horizon = int(z0.time_months.iloc[0])
    ax.plot([0, 1], [0, 1], color="#AEB7BE", lw=1.0, ls="--")
    for model in MODELS:
        z = z0[z0.model == model].sort_values("predicted_event_probability")
        ax.plot(z.predicted_event_probability, z.observed_event_probability,
                color=MODEL_COLORS[model], lw=1.4, marker="o", markersize=2.9, clip_on=True)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(f"Predicted risk at {horizon} months")
    ax.set_ylabel(f"Observed risk at {horizon} months")
    style_axis(ax, "both")
    handles = [Line2D([0], [0], color="#AEB7BE", lw=1, ls="--", label="Ideal")] + model_handles
    legend_box(ax, handles, loc="lower right", fontsize=5.8, ncol=1)


def draw_dca(ax, cohort, subtype):
    z = dca[(dca.cohort == cohort) & (dca.subtype == subtype)]
    horizon = int(z.time_months.iloc[0])
    for model in MODELS:
        q = z[z.strategy == model]
        ax.plot(q.threshold, q.net_benefit, color=MODEL_COLORS[model], lw=1.35)
    q = z[z.strategy == "Treat all"]
    ax.plot(q.threshold, q.net_benefit, color="#9AA3AA", lw=1.0, ls="--")
    ax.axhline(0, color="#303840", lw=0.9, ls=":")
    vals = z.net_benefit.to_numpy(float)
    lo = max(-0.10, np.nanpercentile(vals, 1) - 0.02)
    hi = min(0.90, max(0.12, np.nanpercentile(vals, 99) + 0.03))
    ax.set_xlim(0.10, 0.70)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"Threshold probability ({horizon}-month PFS)")
    ax.set_ylabel("Net benefit")
    style_axis(ax, "both")
    handles = [Line2D([0], [0], color=MODEL_COLORS[m], lw=1.4, label=m) for m in MODELS]
    handles += [
        Line2D([0], [0], color="#9AA3AA", lw=1, ls="--", label="Treat all"),
        Line2D([0], [0], color="#303840", lw=1, ls=":", label="Treat none"),
    ]
    legend_box(ax, handles, loc="upper right", fontsize=5.6, ncol=1)


def draw_brier(ax, cohort, subtype):
    z0 = brier[(brier.cohort == cohort) & (brier.subtype == subtype)]
    handles = []
    for model in MODELS:
        z = z0[z0.model == model].sort_values("time_months")
        ax.plot(z.time_months, z.brier, color=MODEL_COLORS[model], lw=1.45,
                marker="o", markersize=2.9, clip_on=True)
        ibs = float(z.ibs.iloc[0])
        handles.append(Line2D(
            [0], [0], color=MODEL_COLORS[model], marker="o", lw=1.6,
            markersize=3.8, label=f"{model} (IBS {ibs:.3f})"
        ))
    vals = z0.brier.to_numpy(float)
    ax.set_xlim(5.8, 36.2)
    ax.set_ylim(max(0, np.nanmin(vals) - 0.025), min(0.32, np.nanmax(vals) + 0.035))
    ax.set_xticks([6, 12, 18, 24, 30, 36])
    ax.set_xlabel("Time since surgery (months)")
    ax.set_ylabel("IPCW Brier score")
    style_axis(ax, "both")
    legend_loc = "upper right" if subtype == "GBM" else "lower right"
    legend_box(ax, handles, loc=legend_loc, fontsize=5.6, ncol=1)


def save_diagnostic_components(metric_folder, drawer):
    for subtype in SUBTYPES:
        cohorts = ALL_COHORTS if subtype == "GBM" else MAIN_COHORTS
        for cohort in cohorts:
            if cohort == "GBM multi-omics subset" and subtype != "GBM":
                continue
            fig, ax = plt.subplots(figsize=(3.45, 2.45))
            fig.subplots_adjust(left=0.19, right=0.985, top=0.975, bottom=0.23)
            drawer(ax, cohort, subtype)
            save_component(fig, metric_folder, f"{SUBTYPE_SLUG[subtype]}_{COHORT_SLUG[cohort]}.png")


pair_lookup = {}
for row in comp.itertuples():
    a, b = row.comparison.split(" vs ")
    pair_lookup[(row.cohort, row.subtype, row.metric, a, b)] = row.p_value
    pair_lookup[(row.cohort, row.subtype, row.metric, b, a)] = row.p_value

cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "pvalue_teal", ["#173E52", "#4F8E91", "#A7CECA", "#F4F7F7"]
)
norm = mpl.colors.LogNorm(vmin=0.001, vmax=1.0)


def draw_pair_matrix(ax, cohort, subtype, metric):
    row_models = MODELS[1:]
    col_models = MODELS[:-1]
    for r, comparison_model in enumerate(row_models):
        for c, reference_model in enumerate(col_models):
            if MODELS.index(comparison_model) <= MODELS.index(reference_model):
                continue
            p = pair_lookup[(cohort, subtype, metric, comparison_model, reference_model)]
            color = cmap(norm(max(0.001, p)))
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor=color,
                                   edgecolor="#C8D4DA", lw=0.75))
            label = "<0.001" if p < 0.001 else f"{p:.3f}"
            ax.text(c + 0.5, r + 0.5, label, ha="center", va="center",
                    color="white" if p < 0.015 else DARK, fontsize=8.0)
    ax.set_xlim(0, 3)
    ax.set_ylim(3, 0)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(3) + 0.5, col_models, rotation=38, ha="right")
    ax.set_yticks(np.arange(3) + 0.5, row_models)
    ax.tick_params(length=0, pad=1.2, labelsize=7.4)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_bridge_difference(ax, metric):
    groups = [(cohort, subtype) for cohort in MAIN_COHORTS[1:] for subtype in SUBTYPES]
    rows = []
    for cohort, subtype in groups:
        hit = comp[
            (comp.cohort == cohort)
            & (comp.subtype == subtype)
            & (comp.metric == metric)
            & (comp.comparison == "CR-ViT vs 3D-ViT")
        ]
        rows.append(hit.iloc[0])
    y = np.arange(len(rows))[::-1]
    estimate = np.array([row.difference for row in rows], dtype=float)
    lower = np.array([row.ci_low for row in rows], dtype=float)
    upper = np.array([row.ci_high for row in rows], dtype=float)
    ax.errorbar(
        estimate,
        y,
        xerr=[estimate - lower, upper - estimate],
        fmt="o",
        color=MODEL_COLORS["CR-ViT"],
        capsize=2.5,
        lw=1.25,
        markersize=5.5,
    )
    ax.axvline(0, color=DARK, lw=0.9)
    ax.set_yticks(
        y,
        [f"{'Temporal' if cohort.startswith('Temporal') else 'Spatial'} | {subtype}"
         for cohort, subtype in groups],
    )
    label = "Difference in C-index (CR-ViT – 3D-ViT)" if metric == "cindex" else \
        "Difference in iAUC (CR-ViT – 3D-ViT)"
    ax.set_xlabel(label)
    style_axis(ax, "x")


def draw_bridge_associations(ax):
    matrix = bridge.pivot(index="factor", columns="subtype", values="association")
    order = matrix.abs().max(axis=1).sort_values(ascending=False).index[:10]
    matrix = matrix.loc[order, SUBTYPES]
    image = ax.imshow(matrix.values, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_xticks(range(3), [SUBTYPE_LABELS[subtype] for subtype in SUBTYPES], rotation=25, ha="right")
    ax.set_yticks(range(len(matrix)), matrix.index)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix.iat[row, col]
            shown = 0.0 if np.isfinite(value) and abs(value) < 0.005 else value
            ax.text(
                col,
                row,
                "—" if not np.isfinite(shown) else f"{shown:.2f}",
                ha="center",
                va="center",
                fontsize=7.2,
                color="white" if np.isfinite(value) and abs(value) > 0.35 else DARK,
            )
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.040, pad=0.015)
    colorbar.ax.set_title("Assoc.", fontsize=7.5, fontweight="bold", pad=3)
    colorbar.ax.tick_params(labelsize=7.0)


def draw_bridge_residual(ax):
    frame = (
        table3_cox[table3_cox.population == "Pooled validation cohorts"]
        .set_index("subtype")
        .loc[SUBTYPES]
        .reset_index()
    )
    y = np.arange(len(frame))[::-1]
    ax.errorbar(
        frame.adjusted_hr_per_training_sd_vit,
        y,
        xerr=[
            frame.adjusted_hr_per_training_sd_vit - frame.ci_low,
            frame.ci_high - frame.adjusted_hr_per_training_sd_vit,
        ],
        fmt="o",
        color=MODEL_COLORS["3D-ViT"],
        capsize=2.5,
        lw=1.25,
        markersize=5.5,
    )
    ax.axvline(1, color=DARK, lw=0.9)
    ax.set_yticks(y, [SUBTYPE_LABELS[subtype] for subtype in frame.subtype])
    ax.set_xlabel("Adjusted HR per 1-SD ViT score")
    style_axis(ax, "x")
    right = ax.get_xlim()[1]
    for yi, row in zip(y, frame.itertuples()):
        r2 = bridge_r2.loc[
            bridge_r2.subtype == row.subtype, "cross_validated_r2"
        ].iloc[0]
        ax.text(right, yi, f"CV R² = {r2:.3f}", ha="right", va="bottom", fontsize=7.5)


def main():
    for cohort in MAIN_COHORTS:
        fig, ax = plt.subplots(figsize=(3.9, 2.35))
        fig.subplots_adjust(left=0.16, right=0.985, top=0.96, bottom=0.20)
        draw_cindex_by_subtype(ax, cohort)
        save_component(fig, "Figure3", f"A_{COHORT_SLUG[cohort]}.png")

    for subtype in SUBTYPES:
        for cohort in MAIN_COHORTS:
            loc = "upper left" if subtype != "GBM" else "lower right"
            fig, ax = plt.subplots(figsize=(4.20, 1.85))
            fig.subplots_adjust(left=0.17, right=0.985, top=0.96, bottom=0.20)
            draw_auc(ax, cohort, subtype, loc)
            save_component(fig, "Figure3", f"B_{SUBTYPE_SLUG[subtype]}_{COHORT_SLUG[cohort]}.png")

    for subtype in SUBTYPES:
        for cohort in MAIN_COHORTS:
            frame = pred[(pred.cohort == cohort) & (pred.subtype == subtype)].copy()
            stat = km_stats[(km_stats.cohort == cohort) & (km_stats.subtype == subtype)].iloc[0]
            loc = "lower left" if subtype != "GBM" else "upper right"
            draw_km_component(
                frame, stat, "Figure3",
                f"C_{SUBTYPE_SLUG[subtype]}_{COHORT_SLUG[cohort]}.png",
                "crvit_risk_group", loc
            )

    fig, ax = plt.subplots(figsize=(4.20, 2.30))
    fig.subplots_adjust(left=0.18, right=0.985, top=0.96, bottom=0.25)
    draw_cindex_gbm_multiomics(ax)
    save_component(fig, "Figure3", "D_gbm_multiomics_cindex.png")

    fig, ax = plt.subplots(figsize=(4.20, 2.30))
    fig.subplots_adjust(left=0.18, right=0.985, top=0.96, bottom=0.22)
    draw_auc(ax, "GBM multi-omics subset", "GBM", "lower right")
    save_component(fig, "Figure3", "D_gbm_multiomics_auc.png")

    frame = pred[(pred.patient_id.isin(multi_ids)) & (pred.subtype == "GBM")].copy()
    stat = km_stats[(km_stats.cohort == "GBM multi-omics subset") & (km_stats.subtype == "GBM")].iloc[0]
    draw_km_component(frame, stat, "Figure3", "D_gbm_multiomics_km.png", "crvit_risk_group", "upper right")

    save_diagnostic_components("S2_calibration", draw_calibration)
    save_diagnostic_components("S3_dca", draw_dca)
    save_diagnostic_components("S4_brier", draw_brier)

    for subtype in SUBTYPES:
        for cohort in MAIN_COHORTS:
            frame = pred[(pred.cohort == cohort) & (pred.subtype == subtype)].copy()
            stat = vit_km_stats[(vit_km_stats.cohort == cohort) & (vit_km_stats.subtype == subtype)].iloc[0]
            loc = "lower left" if subtype != "GBM" else "upper right"
            draw_km_component(
                frame, stat, "S6_vit_km",
                f"{SUBTYPE_SLUG[subtype]}_{COHORT_SLUG[cohort]}.png",
                "vit_risk_group", loc
            )
    frame = pred[(pred.patient_id.isin(multi_ids)) & (pred.subtype == "GBM")].copy()
    stat = vit_km_stats[(vit_km_stats.cohort == "GBM multi-omics subset") & (vit_km_stats.subtype == "GBM")].iloc[0]
    draw_km_component(frame, stat, "S6_vit_km", "gbm_multiomics.png", "vit_risk_group", "upper right")

    for metric in ["cindex", "iauc"]:
        for subtype in SUBTYPES:
            for cohort in MAIN_COHORTS:
                fig, ax = plt.subplots(figsize=(2.25, 2.05))
                fig.subplots_adjust(left=0.24, right=0.98, top=0.96, bottom=0.25)
                draw_pair_matrix(ax, cohort, subtype, metric)
                save_component(
                    fig, "S5_pairwise",
                    f"{metric}_{SUBTYPE_SLUG[subtype]}_{COHORT_SLUG[cohort]}.png",
                    tight=True,
                )
        fig, ax = plt.subplots(figsize=(2.25, 2.05))
        fig.subplots_adjust(left=0.24, right=0.98, top=0.96, bottom=0.25)
        draw_pair_matrix(ax, "GBM multi-omics subset", "GBM", metric)
        save_component(fig, "S5_pairwise", f"{metric}_gbm_multiomics.png", tight=True)

    print(f"Wrote PPT-ready components to {OUT}")


if __name__ == "__main__":
    main()
