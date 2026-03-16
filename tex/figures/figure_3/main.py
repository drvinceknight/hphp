from __future__ import annotations

import pathlib

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# rcParams
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial"],
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.0,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 300,
    }
)

MALE_COLOR = "#0072B2"
FEMALE_COLOR = "#CC79A7"
TRAJ_COLOR = "#2A9D8F"

PALETTE = [
    "#0072B2", "#E69F00", "#009E73", "#CC79A7",
    "#56B4E9", "#D55E00", "#F0E442", "#000000",
]
LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]

PYRAMID_YEARS = [0, 50, 100, 150, 200]
PYRAMID_BIN_SIZE = 5  # group single-year ages into 5-year bins for display
AGE_DISPLAY_BINS = list(range(0, 100, PYRAMID_BIN_SIZE))
AGE_LABELS = [f"{a}–{a+4}" for a in AGE_DISPLAY_BINS[:-1]] + ["95+"]

# Life-table death probabilities (UN 1985, from birth_death.py)
_MALE_Q = np.array([
    0.07446, 0.01592, 0.00925, 0.00641, 0.00485, 0.00377, 0.00299,
    0.00241, 0.00199, 0.00172, 0.00157, 0.00150, 0.00150, 0.00153,
    0.00161, 0.00173, 0.00189, 0.00209, 0.00230, 0.00248, 0.00265,
    0.00279, 0.00290, 0.00303, 0.00311, 0.00312, 0.00310, 0.00309,
    0.00313, 0.00317, 0.00326, 0.00334, 0.00345, 0.00356, 0.00368,
    0.00385, 0.00400, 0.00407, 0.00433, 0.00455, 0.00488, 0.00516,
    0.00550, 0.00596, 0.00634, 0.00686, 0.00736, 0.00785, 0.00848,
    0.00909, 0.00992, 0.01064, 0.01154, 0.01240, 0.01327, 0.01428,
    0.01534, 0.01660, 0.01805, 0.01963, 0.02168, 0.02369, 0.02580,
    0.02821, 0.03048, 0.03334, 0.03566, 0.03874, 0.04193, 0.04536,
    0.04968, 0.05367, 0.05856, 0.06368, 0.06924, 0.07541, 0.08165,
    0.08858, 0.09608, 0.10417, 0.11372, 0.12342, 0.13357, 0.14442,
    0.15646, 0.16863, 0.17992, 0.19350, 0.20749, 0.22131, 0.23581,
    0.25014, 0.26705, 0.28080, 0.29747, 0.31033, 0.32555, 0.34112,
    0.35664, 0.37224, 1.0000,
])

_FEMALE_Q = np.array([
    0.07033, 0.01503, 0.00883, 0.00630, 0.00494, 0.00393, 0.00312,
    0.00246, 0.00194, 0.00159, 0.00141, 0.00132, 0.00130, 0.00131,
    0.00137, 0.00145, 0.00155, 0.00166, 0.00177, 0.00183, 0.00188,
    0.00192, 0.00197, 0.00205, 0.00211, 0.00214, 0.00216, 0.00217,
    0.00222, 0.00226, 0.00231, 0.00236, 0.00242, 0.00247, 0.00254,
    0.00264, 0.00280, 0.00288, 0.00306, 0.00326, 0.00346, 0.00360,
    0.00376, 0.00393, 0.00409, 0.00433, 0.00459, 0.00488, 0.00530,
    0.00571, 0.00622, 0.00670, 0.00720, 0.00766, 0.00811, 0.00865,
    0.00927, 0.01002, 0.01096, 0.01199, 0.01329, 0.01457, 0.01605,
    0.01767, 0.01921, 0.02124, 0.02314, 0.02537, 0.02758, 0.02976,
    0.03265, 0.03525, 0.03877, 0.04252, 0.04665, 0.05145, 0.05614,
    0.06161, 0.06740, 0.07378, 0.08176, 0.08972, 0.09839, 0.10751,
    0.11900, 0.13060, 0.14029, 0.15284, 0.16478, 0.17836, 0.19303,
    0.20629, 0.22322, 0.23829, 0.25346, 0.27026, 0.28709, 0.30468,
    0.32225, 0.33985, 1.0000,
])


def _survivorship(q):
    """Compute l(x) from age-specific death probabilities. l(0) = 1."""
    l = np.ones(len(q))
    for i in range(1, len(q)):
        l[i] = l[i - 1] * (1.0 - q[i - 1])
    return l


# Data loading

def load_data(raw_data_path, q_heal=None, alpha=None):
    """
    Load summary and pyramid CSVs from a single representative parameter
    directory within raw_data_path.

    If `q_heal` and `alpha` are given, the directory whose name matches
    ``heal{q_heal}`` and ``alpha{alpha}`` is selected (trauma0.0 preferred
    when multiple matches exist).  Otherwise the first no-trauma directory
    with the smallest alpha value is used as a baseline.

    Only directories where both main.csv and pyramids.csv are present are
    considered.
    """
    dirs = sorted(raw_data_path.glob("*/"))
    complete = [
        d for d in dirs
        if (d / "main.csv").exists() and (d / "pyramids.csv").exists()
    ]
    if not complete:
        raise FileNotFoundError(
            f"No completed runs (with both main.csv and pyramids.csv) found in "
            f"{raw_data_path}."
        )

    if q_heal is not None and alpha is not None:
        heal_tag = f"heal{q_heal}"
        alpha_tag = f"alpha{alpha}"
        candidates = [
            d for d in complete
            if heal_tag in d.name and alpha_tag in d.name
        ]
        if not candidates:
            raise FileNotFoundError(
                f"No completed run found for alpha={alpha}, q_heal={q_heal} "
                f"in {raw_data_path}."
            )
        # Prefer no-trauma run when multiple matches
        no_trauma = [d for d in candidates if "trauma0.0" in d.name]
        data_dir = no_trauma[0] if no_trauma else candidates[0]
    else:
        # Prefer a no-trauma run; fall back to first complete directory
        candidates = [d for d in complete if "trauma0.0" in d.name]
        data_dir = candidates[0] if candidates else complete[0]

    summary_df = pd.read_csv(data_dir / "main.csv")
    pyramid_df = pd.read_csv(data_dir / "pyramids.csv")
    return summary_df, pyramid_df


# Panel helpers

def _traj_panel(ax, df, col, ylabel, label, subsets, p_trauma, warm_up=0):
    """Median ± IQR for each (q, alpha) subset."""
    df_p = df[df["probability_of_trauma"] == p_trauma]
    for i, (q, alpha) in enumerate(subsets):
        mask = (df_p["probability_of_heal"] == q) & (df_p["alpha"] == alpha)
        grp = df_p[mask].groupby("year")[col]
        med = grp.median()
        q25 = grp.quantile(0.25)
        q75 = grp.quantile(0.75)
        color = PALETTE[i % len(PALETTE)]
        ls = LINESTYLES[i % len(LINESTYLES)]
        ax.plot(med.index, med.values, color=color, linestyle=ls,
                label=f"q={q}, α={alpha}")
        ax.fill_between(med.index, q25.values, q75.values,
                        color=color, alpha=0.18, linewidth=0)

    if warm_up > 0:
        ax.axvline(warm_up, color="black", linewidth=0.8, linestyle=":", zorder=5)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_title(f"$\\mathbf{{{label}}}$", loc="left")
    ax.spines[["top", "right"]].set_visible(False)


def _sex_traj_panel(ax, df, label, subsets, p_trauma, warm_up=0):
    """Median males and females for each (q, alpha) subset."""
    df_p = df[df["probability_of_trauma"] == p_trauma]
    for i, (q, alpha) in enumerate(subsets):
        mask = (df_p["probability_of_heal"] == q) & (df_p["alpha"] == alpha)
        sub = df_p[mask]
        ls = LINESTYLES[i % len(LINESTYLES)]
        for col, color in [("males", MALE_COLOR), ("females", FEMALE_COLOR)]:
            grp = sub.groupby("year")[col]
            med = grp.median()
            ax.plot(med.index, med.values, color=color, linestyle=ls, linewidth=1.0)

    if warm_up > 0:
        ax.axvline(warm_up, color="black", linewidth=0.8, linestyle=":", zorder=5)
    handles = [
        Line2D([0], [0], color=MALE_COLOR, linewidth=1.4, label="Males"),
        Line2D([0], [0], color=FEMALE_COLOR, linewidth=1.4, label="Females"),
    ]
    ax.legend(handles=handles, frameon=False, handlelength=1.2, borderpad=0,
              loc="upper left", borderaxespad=0.2)
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    ax.set_title(f"$\\mathbf{{{label}}}$", loc="left")
    ax.spines[["top", "right"]].set_visible(False)


def _pooled_empirical_survivorship(pyramid_dfs, warm_up=100):
    """
    Compute age-wise IQR of l(x) pooled across all pyramid DataFrames and
    all replicates/years >= warm_up.  Returns (ages, q25_male, q75_male,
    q25_female, q75_female).
    """
    all_lm, all_lf = [], []
    for pyramid_df in pyramid_dfs:
        if pyramid_df is None:
            continue
        sub = pyramid_df[pyramid_df["year"] >= warm_up].copy()
        base = sub[sub["age_group"] == 0][["rep", "year", "males", "females"]].rename(
            columns={"males": "m0", "females": "f0"}
        )
        merged = sub.merge(base, on=["rep", "year"])
        merged["l_male"] = merged["males"] / merged["m0"].replace(0, np.nan)
        merged["l_female"] = merged["females"] / merged["f0"].replace(0, np.nan)
        all_lm.append(merged[["age_group", "l_male"]])
        all_lf.append(merged[["age_group", "l_female"]])

    if not all_lm:
        return None
    lm_df = pd.concat(all_lm)
    lf_df = pd.concat(all_lf)
    ages = np.array(sorted(lm_df["age_group"].unique()))
    q25_m = lm_df.groupby("age_group")["l_male"].quantile(0.25).reindex(ages).values
    q75_m = lm_df.groupby("age_group")["l_male"].quantile(0.75).reindex(ages).values
    q25_f = lf_df.groupby("age_group")["l_female"].quantile(0.25).reindex(ages).values
    q75_f = lf_df.groupby("age_group")["l_female"].quantile(0.75).reindex(ages).values
    return ages, q25_m, q75_m, q25_f, q75_f


def _empirical_survivorship(pyramid_df, warm_up=100):
    """
    Compute period survivorship l(a) = N(a) / N(0) per sex from simulation
    data, using snapshot years >= warm_up.  Returns (ages, l_male, l_female)
    as median ± IQR arrays across reps and years.
    """
    sub = pyramid_df[pyramid_df["year"] >= warm_up].copy()
    # For each (rep, year), normalise counts by age-0 count
    base = sub[sub["age_group"] == 0][["rep", "year", "males", "females"]].rename(
        columns={"males": "m0", "females": "f0"}
    )
    merged = sub.merge(base, on=["rep", "year"])
    merged["l_male"] = merged["males"] / merged["m0"].replace(0, np.nan)
    merged["l_female"] = merged["females"] / merged["f0"].replace(0, np.nan)

    grp = merged.groupby("age_group")
    ages = np.array(sorted(merged["age_group"].unique()))
    l_male_med = grp["l_male"].median().reindex(ages).values
    l_male_q25 = grp["l_male"].quantile(0.25).reindex(ages).values
    l_male_q75 = grp["l_male"].quantile(0.75).reindex(ages).values
    l_female_med = grp["l_female"].median().reindex(ages).values
    l_female_q25 = grp["l_female"].quantile(0.25).reindex(ages).values
    l_female_q75 = grp["l_female"].quantile(0.75).reindex(ages).values
    return (ages,
            l_male_med, l_male_q25, l_male_q75,
            l_female_med, l_female_q25, l_female_q75)


def _survival_panel(ax, label, pyramid_dfs, subsets, warm_up=100):
    """Pooled IQR band + per-subset median survivorship lines."""
    ages_th = np.arange(len(_MALE_Q))

    # Pooled empirical IQR band (grey)
    pooled = _pooled_empirical_survivorship(pyramid_dfs, warm_up=warm_up)
    if pooled is not None:
        ages_e, q25_m, q75_m, q25_f, q75_f = pooled
        ax.fill_between(ages_e, q25_m, q75_m, color="0.75", alpha=0.5,
                        linewidth=0, label="IQR (pooled)")
        ax.fill_between(ages_e, q25_f, q75_f, color="0.75", alpha=0.5,
                        linewidth=0)

    # Theoretical — dark grey reference lines
    ax.plot(ages_th, _survivorship(_MALE_Q), color="0.35",
            linewidth=1.0, linestyle="-", label="Male (theoretical)")
    ax.plot(ages_th, _survivorship(_FEMALE_Q), color="0.35",
            linewidth=1.0, linestyle="--", label="Female (theoretical)")

    # Per-subset median lines only (no per-subset IQR bands)
    for i, ((q, alpha), pyramid_df) in enumerate(zip(subsets, pyramid_dfs)):
        if pyramid_df is None:
            continue
        ages_e, lm, _, _, lf, _, _ = _empirical_survivorship(
            pyramid_df, warm_up=warm_up
        )
        color = PALETTE[i % len(PALETTE)]
        ls = LINESTYLES[i % len(LINESTYLES)]
        ax.plot(ages_e, lm, color=color, linewidth=1.0, linestyle=ls,
                label=f"q={q}, α={alpha}")
        ax.plot(ages_e, lf, color=color, linewidth=1.0, linestyle=ls)

    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Survivorship $l(x)$")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"$\\mathbf{{{label}}}$  [years $\\geq$ {warm_up}]", loc="left")
    ax.legend(frameon=False, handlelength=1.5, borderpad=0,
              loc="lower left", borderaxespad=0.2, fontsize=5)
    ax.spines[["top", "right"]].set_visible(False)


def _pyramid_panel(ax, pyramid_df, year, x_max, label, show_ylabel=True):
    """Horizontal-bar population pyramid for a single snapshot year."""
    sub = pyramid_df[pyramid_df["year"] == year].copy()
    # Aggregate single-year ages into 5-year display bins, then median across reps
    sub["display_bin"] = (sub["age_group"] // PYRAMID_BIN_SIZE) * PYRAMID_BIN_SIZE
    binned = sub.groupby(["rep", "display_bin"])[["males", "females"]].sum().reset_index()
    agg = binned.groupby("display_bin")[["males", "females"]].median().reset_index()
    agg = agg.sort_values("display_bin")

    y_pos = np.arange(len(agg))
    ax.barh(y_pos, -agg["males"].values, color=MALE_COLOR,
            alpha=0.75, linewidth=0)
    ax.barh(y_pos, agg["females"].values, color=FEMALE_COLOR,
            alpha=0.75, linewidth=0)

    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-0.5, len(agg) - 0.5)

    tick_idx = np.arange(0, len(AGE_DISPLAY_BINS), 2)
    ax.set_yticks(tick_idx)
    if show_ylabel:
        ax.set_yticklabels([AGE_LABELS[i] for i in tick_idx], fontsize=5)
        ax.set_ylabel("Age group", fontsize=6)
    else:
        ax.set_yticklabels([])

    x_ticks = np.linspace(0, x_max, 3)
    all_ticks = np.concatenate([-x_ticks[1:][::-1], x_ticks])
    ax.set_xticks(all_ticks)
    ax.set_xticklabels(
        [f"{int(abs(v)/1000)}k" if abs(v) >= 1000 else str(int(abs(v)))
         for v in all_ticks],
        fontsize=5,
    )

    ax.axvline(0, color="black", linewidth=0.5, zorder=5)
    ax.set_xlabel("Count", fontsize=6)
    ax.set_title(f"$\\mathbf{{{label}}}$  Year {year}", loc="left", fontsize=6)

    if show_ylabel:
        ax.text(-x_max * 0.98, len(agg) - 1.5, "Males",
                color=MALE_COLOR, fontsize=5, ha="left", va="top")
        ax.text(x_max * 0.98, len(agg) - 1.5, "Females",
                color=FEMALE_COLOR, fontsize=5, ha="right", va="top")

    ax.spines[["top", "right"]].set_visible(False)


# Figure assembly

def make_figure(summary_df, pyramid_df, pyramid_dfs, subsets,
               rep_q, rep_alpha, p_trauma=0.0, warm_up=100):
    fig = plt.figure(figsize=(15, 8))

    gs_top = gridspec.GridSpec(
        1, 3, figure=fig,
        left=0.07, right=0.97, top=0.95, bottom=0.55,
        wspace=0.38,
    )
    gs_bot = gridspec.GridSpec(
        1, 5, figure=fig,
        left=0.07, right=0.97, top=0.44, bottom=0.06,
        wspace=0.10,
    )

    ax_N = fig.add_subplot(gs_top[0, 0])
    ax_age = fig.add_subplot(gs_top[0, 1])
    ax_surv = fig.add_subplot(gs_top[0, 2])
    ax_pyrs = [fig.add_subplot(gs_bot[0, i]) for i in range(5)]

    # Row 1 — one line per (q, alpha) subset
    _traj_panel(ax_N, summary_df, "population_size", "Population size", "A",
                subsets, p_trauma, warm_up=warm_up)
    _traj_panel(ax_age, summary_df, "mean_age", "Mean age (years)", "B",
                subsets, p_trauma, warm_up=warm_up)
    _survival_panel(ax_surv, "C", pyramid_dfs, subsets, warm_up=warm_up)

    # Legends for trajectory panels A and B
    handles, labels = ax_N.get_legend_handles_labels()
    ax_N.legend(handles, labels, frameon=False, handlelength=1.2,
                title="q, α", title_fontsize=6, fontsize=6,
                loc="upper left", borderaxespad=0.2)
    ax_age.legend(handles, labels, frameon=False, handlelength=1.2,
                  title="q, α", title_fontsize=6, fontsize=6,
                  loc="upper right", borderaxespad=0.2)

    # Row 2 — population pyramids from the representative run
    rep_str = f"q = {rep_q}, α = {rep_alpha}"
    x_max_vals = []
    for year in PYRAMID_YEARS:
        sub = pyramid_df[pyramid_df["year"] == year].copy()
        sub["display_bin"] = (sub["age_group"] // PYRAMID_BIN_SIZE) * PYRAMID_BIN_SIZE
        binned = sub.groupby(["rep", "display_bin"])[["males", "females"]].sum().reset_index()
        agg = binned.groupby("display_bin")[["males", "females"]].median()
        x_max_vals.append(max(agg["males"].max(), agg["females"].max()))
    x_max = float(max(x_max_vals)) * 1.12

    for i, (year, ax) in enumerate(zip(PYRAMID_YEARS, ax_pyrs)):
        _pyramid_panel(
            ax, pyramid_df, year, x_max,
            label="DEFGH"[i],
            show_ylabel=(i == 0),
        )
    # Label the representative run above the first pyramid panel
    ax_pyrs[0].set_title(
        f"$\\mathbf{{D}}$  Year {PYRAMID_YEARS[0]}  [{rep_str}]",
        loc="left", fontsize=6,
    )

    return fig


# Entry point

if __name__ == "__main__":
    raw_data_path = pathlib.Path(__file__).parent.parent / "figure_4" / "data" / "raw"

    # Same trajectory subsets and p_high as figure 4
    trajectory_subsets = [
        (0.0, 1.35),
        (0.0, 1.05),
        (0.01, 1.35),
        (0.01, 1.05),
        (0.05, 1.35),
        (0.05, 1.05),
    ]
    p_trauma = 0.0
    warm_up = 100

    # Load all summary CSVs (same as figure 4)
    csv_paths = list(raw_data_path.glob("**/main.csv"))
    summary_df = pd.concat(
        pd.read_csv(p) for p in csv_paths if p.stat().st_size > 0
    )

    # Load pyramid data for each subset (for survivorship panel)
    pyramid_dfs = []
    for q, alpha in trajectory_subsets:
        try:
            _, pdf = load_data(raw_data_path, q_heal=q, alpha=alpha)
            pyramid_dfs.append(pdf)
        except FileNotFoundError:
            pyramid_dfs.append(None)

    # Representative run for the population pyramids
    rep_q, rep_alpha = 0.05, 1.35
    _, pyramid_df = load_data(raw_data_path, q_heal=rep_q, alpha=rep_alpha)

    fig = make_figure(summary_df, pyramid_df, pyramid_dfs, trajectory_subsets,
                      rep_q=rep_q, rep_alpha=rep_alpha,
                      p_trauma=p_trauma, warm_up=warm_up)
    out = pathlib.Path(__file__).parent / "main.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
