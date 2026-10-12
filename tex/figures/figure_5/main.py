from __future__ import annotations

import pathlib

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cluster import KMeans

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

# Colour / style cycles (Wong palette, colourblind-safe)
PALETTE = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#F0E442",
    "#000000",
]
LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "ace_heat", ["#FFFFFF", "#A8D8D8", "#2A9D8F", "#264653"], N=256
)


# K-means threshold
def _kmeans_threshold(values):
    """Midpoint between two k-means (k=2) cluster centres on a flat finite array."""
    flat = values.flatten()
    flat = flat[np.isfinite(flat)]
    km = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=0)
    km.fit(flat[:, None])
    c0, c1 = sorted(km.cluster_centers_.flatten())
    return (c0 + c1) / 2


def _pooled_kmeans_threshold(df_summary, metric, p_low, p_high):
    """
    k-means threshold on log1p(metric), pooled over the two exposure levels
    shown in the main figure.

    Both the main heatmaps and the SI contour figure cut at this level, so the
    threshold heal rate q* they report for a given p is the same number.
    """
    pooled = []
    for p_trauma in (p_low, p_high):
        sub = df_summary[df_summary["probability_of_trauma"] == p_trauma]
        values = sub.groupby(["probability_of_heal", "alpha"])[metric].mean().values
        pooled.append(np.log1p(values.flatten()))
    return _kmeans_threshold(np.concatenate(pooled))


def _threshold_crossings(pivot, threshold_log):
    """
    For each alpha column of the pivot, find the heal rate q at which
    log1p(metric) crosses threshold_log, by linear interpolation between the
    two bracketing grid values.

    Returns (alphas, crossings) as arrays over the columns that cross.
    """
    log_values = np.log1p(pivot.values)
    heal_values = pivot.index.values.astype(float)
    alpha_values = pivot.columns.values.astype(float)

    alphas = []
    crossings = []
    for column, alpha in enumerate(alpha_values):
        profile = log_values[:, column]
        for row in range(len(heal_values) - 1):
            below, above = profile[row], profile[row + 1]
            if (below - threshold_log) * (above - threshold_log) <= 0:
                span = above - below
                if abs(span) < 1e-12:
                    crossing = heal_values[row]
                else:
                    crossing = heal_values[row] + (threshold_log - below) / span * (
                        heal_values[row + 1] - heal_values[row]
                    )
                alphas.append(alpha)
                crossings.append(crossing)
                break

    return np.array(alphas), np.array(crossings)


def _kmeans_linear_fit(pivot, threshold_log):
    """
    Fit q* = slope * alpha + intercept by OLS to the threshold contour.

    Returns (slope, intercept, r_squared).
    Returns None if the contour does not cross in at least two columns.
    """
    alphas, crossings = _threshold_crossings(pivot, threshold_log)
    if len(alphas) < 2:
        return None

    design = np.column_stack([alphas, np.ones_like(alphas)])
    slope, intercept = np.linalg.lstsq(design, crossings, rcond=None)[0]
    residual_sum = ((crossings - (slope * alphas + intercept)) ** 2).sum()
    total_sum = ((crossings - crossings.mean()) ** 2).sum()
    r_squared = 1 - residual_sum / total_sum if total_sum > 1e-12 else 1.0
    return slope, intercept, r_squared


# Panel drawing functions


def _panel_trajectories(
    ax, df, column, ylabel, panel_label, subsets, p_trauma, warm_up=0, show_legend=True
):
    """Median ± IQR of `column` over time for each (q_heal, alpha) subset."""
    df_p = df[df["probability_of_trauma"] == p_trauma]
    for i, (q, alpha) in enumerate(subsets):
        mask = (df_p["probability_of_heal"] == q) & (df_p["alpha"] == alpha)
        grp = df_p[mask].groupby("year")[column]
        med = grp.median()
        q25 = grp.quantile(0.25)
        q75 = grp.quantile(0.75)
        col = PALETTE[i % len(PALETTE)]
        ls = LINESTYLES[i % len(LINESTYLES)]
        mk = MARKERS[i % len(MARKERS)]
        n_pts = len(med)
        mark_every = max(1, n_pts // min(5, n_pts))
        ax.plot(
            med.index,
            med.values,
            color=col,
            linestyle=ls,
            marker=mk,
            markersize=3,
            markevery=mark_every,
            label=f"q={q}, \u03b1={alpha}",
        )
        ax.fill_between(
            med.index, q25.values, q75.values, color=col, alpha=0.18, linewidth=0
        )

    if warm_up > 0:
        ax.axvline(warm_up, color="black", linewidth=0.8, linestyle=":", zorder=5)

    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_title(f"$\\mathbf{{{panel_label}}}$  p = {p_trauma}", loc="left")
    if show_legend:
        ax.legend(
            frameon=False,
            handlelength=1.2,
            borderpad=0,
            loc="lower left",
            bbox_to_anchor=(0.01, 0.01),
            borderaxespad=0,
        )
    ax.spines[["top", "right"]].set_visible(False)


def _panel_summary(
    ax,
    df,
    column,
    ylabel,
    panel_label,
    q_values,
    p_trauma,
    show_legend=False,
    wu_note="",
):
    """Mean ± SD of `column` vs alpha, one line per q (heal) value."""
    df_p = df[df["probability_of_trauma"] == p_trauma]
    if q_values is None:
        q_values = sorted(df_p["probability_of_heal"].unique())

    grp = df_p.groupby(["probability_of_heal", "alpha"])[column]
    mn = grp.mean().rename("mean").reset_index()
    sd = grp.std().rename("std").reset_index()
    merged = mn.merge(sd, on=["probability_of_heal", "alpha"])

    for i, q in enumerate(q_values):
        sub = merged[merged["probability_of_heal"] == q].sort_values("alpha")
        col = PALETTE[i % len(PALETTE)]
        ls = LINESTYLES[i % len(LINESTYLES)]
        mk = MARKERS[i % len(MARKERS)]
        n_pts = len(sub)
        if n_pts == 0:
            continue
        mark_every = max(1, n_pts // min(5, n_pts))
        ax.plot(
            sub["alpha"],
            sub["mean"],
            color=col,
            linestyle=ls,
            marker=mk,
            markersize=3,
            markevery=mark_every,
            label=f"q = {q}",
        )
        ax.fill_between(
            sub["alpha"],
            sub["mean"] - sub["std"],
            sub["mean"] + sub["std"],
            color=col,
            alpha=0.15,
            linewidth=0,
        )

    ax.set_xlabel("\u03b1")
    ax.set_ylabel(ylabel)
    ax.set_title(f"$\\mathbf{{{panel_label}}}$  p = {p_trauma}{wu_note}", loc="left")
    if show_legend:
        ax.legend(
            frameon=False,
            title="q",
            title_fontsize=6,
            handlelength=1.2,
            borderpad=0,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
        )
    ax.spines[["top", "right"]].set_visible(False)


def _panel_heatmap(
    ax,
    df,
    column,
    cbar_label,
    panel_label,
    p_trauma,
    vmin,
    vmax,
    threshold_log,
    show_ylabel=True,
    show_colorbar=False,
    wu_note="",
):
    """log1p heatmap of mean `column` with threshold contour and linear-fit title."""
    df_p = df[df["probability_of_trauma"] == p_trauma]
    pivot = df_p.groupby(["probability_of_heal", "alpha"])[column].mean().unstack()
    pivot = pivot.dropna(axis=1)
    log_v = np.log1p(pivot.values)

    im = ax.imshow(
        log_v,
        aspect="auto",
        cmap=HEATMAP_CMAP,
        origin="lower",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )

    n_rows, n_cols = log_v.shape
    if n_rows >= 2 and n_cols >= 2:
        ax.contour(
            np.arange(n_cols),
            np.arange(n_rows),
            log_v,
            levels=[threshold_log],
            colors=["#E63946"],
            linewidths=1.2,
            linestyles="--",
        )

    n_alpha = len(pivot.columns)
    xtick_idx = np.round(np.linspace(0, n_alpha - 1, min(5, n_alpha))).astype(int)
    ax.set_xticks(xtick_idx)
    ax.set_xticklabels(
        [f"{pivot.columns[i]:.2f}" for i in xtick_idx], rotation=45, ha="right"
    )

    n_q = len(pivot.index)
    tick_idx = np.round(np.linspace(0, n_q - 1, min(5, n_q))).astype(int)
    ax.set_yticks(tick_idx)
    ax.set_yticklabels([f"{pivot.index[i]:.2f}" for i in tick_idx], rotation=0)
    ax.yaxis.set_tick_params(pad=2)

    ax.set_xlabel("\u03b1")
    if show_ylabel:
        ax.set_ylabel("q")

    fit = _kmeans_linear_fit(pivot, threshold_log)
    if fit is not None:
        m, c, r2 = fit
        sign = "+" if c >= 0 else "-"
        fit_str = f"threshold: q* = {m:.3f}\u03b1 {sign} {abs(c):.3f}  ($R^2$={r2:.2f})"
    else:
        fit_str = ""

    title = f"$\\mathbf{{{panel_label}}}$  p = {p_trauma}{wu_note}"
    if fit_str:
        title += f"\n{fit_str}"
    ax.set_title(title, loc="left", fontsize=6)

    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(cbar_label, fontsize=6)
        log_ticks = np.linspace(vmin, vmax, 5)
        cb.set_ticks(log_ticks)
        cb.set_ticklabels([f"{np.expm1(t):.2g}" for t in log_ticks])
        cb.ax.tick_params(labelsize=5, width=0.5)
        cb.outline.set_linewidth(0.5)
        cb.ax.axhline(threshold_log, color="#E63946", linewidth=1.2, linestyle="--")
        cb.ax.text(
            1.6,
            threshold_log,
            f"{np.expm1(threshold_log):.2g}",
            transform=cb.ax.get_yaxis_transform(),
            va="center",
            ha="left",
            fontsize=5,
            color="#E63946",
        )


def make_threshold_vs_p_figure(
    df, metric, warm_up=0, p_low=0.0, p_high=0.05, figsize=(7.0, 4.5)
):
    """
    Trace the threshold contour in (alpha, q) space for every value of p.

    The threshold is the one used by the main figure, pooled over p_low and
    p_high, so the contours here are directly comparable in level with the
    heatmap panels.  Values of p whose surface lies wholly above the threshold
    have no crossing and are reported in the caption rather than plotted.
    """
    df_summary = df[df["year"] >= warm_up]
    p_values = sorted(df_summary["probability_of_trauma"].unique())
    threshold_log = _pooled_kmeans_threshold(df_summary, metric, p_low, p_high)

    fig, ax = plt.subplots(figsize=figsize)

    colours = plt.cm.viridis(np.linspace(0.05, 0.95, len(p_values)))

    uncrossed = []
    for index, p_trauma in enumerate(p_values):
        sub = df_summary[df_summary["probability_of_trauma"] == p_trauma]
        pivot = sub.groupby(["probability_of_heal", "alpha"])[metric].mean().unstack()
        pivot = pivot.dropna(axis=1)
        if pivot.shape[0] < 2 or pivot.shape[1] < 2:
            continue

        alphas, crossings = _threshold_crossings(pivot, threshold_log)
        if len(alphas) < 2:
            uncrossed.append(p_trauma)
            continue

        ax.plot(
            alphas,
            crossings,
            color=colours[index],
            linewidth=1.2,
            marker="o",
            markersize=2.5,
            markevery=max(1, len(alphas) // 5),
            label=f"p = {p_trauma:.2g}",
        )

    ax.legend(
        frameon=False,
        title="p",
        title_fontsize=6,
        handlelength=1.5,
        borderpad=0,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        fontsize=6,
    )
    ax.set_xlabel("\u03b1")
    ax.set_ylabel("q* (threshold heal rate)")
    ax.set_title(
        "Threshold heal rate against \u03b1 for each background exposure rate p",
        loc="left",
        fontsize=7,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if uncrossed:
        print(f"  no threshold crossing for p in {uncrossed}")
    return fig


def make_figure_2col(
    df,
    metric,
    metric_ylabel,
    cbar_label,
    q_heal=None,
    p_high=None,
    trajectory_subsets=None,
    warm_up=0,
    adjust_y_limits=False,
    figsize=(11.0, 7.0),
):
    """
    Build a 3-row × 2-column figure for a single outcome metric.

    Columns correspond to p=0 (left) and p=p_high (right).
    Panels are labelled A–F.
    """
    p_low = 0.0
    if p_high is None:
        p_high = sorted(df["probability_of_trauma"].unique())[-1]

    df_summary = df[df["year"] >= warm_up]
    wu_note = f"  [years \u2265 {warm_up}]" if warm_up > 0 else ""

    all_log = []
    for p in [p_low, p_high]:
        sub = df_summary[df_summary["probability_of_trauma"] == p]
        vals = sub.groupby(["probability_of_heal", "alpha"])[metric].mean().values
        all_log.append(np.log1p(vals.flatten()))
    all_log = np.concatenate(all_log)
    all_log = all_log[np.isfinite(all_log)]
    vmin, vmax = all_log.min(), all_log.max()
    threshold_log = _pooled_kmeans_threshold(df_summary, metric, p_low, p_high)

    if trajectory_subsets is None:
        q_vals = sorted(df["probability_of_heal"].unique())
        a_vals = sorted(df["alpha"].unique())
        trajectory_subsets = [
            (q_vals[0], a_vals[0]),
            (q_vals[0], a_vals[-1]),
            (q_vals[-1], a_vals[0]),
            (q_vals[-1], a_vals[-1]),
        ]

    fig = plt.figure(figsize=figsize)

    row_tops = [0.97, 0.65, 0.30]
    row_bottoms = [0.70, 0.38, 0.02]

    axes = []
    for top, bot in zip(row_tops, row_bottoms):
        gs = gridspec.GridSpec(
            1, 2, figure=fig, left=0.12, right=0.88, top=top, bottom=bot, wspace=0.45
        )
        axes.append([fig.add_subplot(gs[0, c]) for c in range(2)])

    ax_A, ax_B = axes[0]
    ax_C, ax_D = axes[1]
    ax_E, ax_F = axes[2]

    # Row 1 — trajectories
    _panel_trajectories(
        ax_A,
        df,
        metric,
        metric_ylabel,
        "A",
        trajectory_subsets,
        p_low,
        warm_up=warm_up,
        show_legend=False,
    )
    _panel_trajectories(
        ax_B,
        df,
        metric,
        metric_ylabel,
        "B",
        trajectory_subsets,
        p_high,
        warm_up=warm_up,
        show_legend=False,
    )

    handles, labels = ax_A.get_legend_handles_labels()
    if warm_up > 0:
        handles.append(Line2D([0], [0], color="black", linewidth=0.8, linestyle=":"))
        labels.append(f"warm-up (year {warm_up})")
    ax_B.legend(
        handles,
        labels,
        frameon=False,
        handlelength=1.5,
        title="q, \u03b1",
        title_fontsize=6,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        fontsize=6,
    )

    # Row 2 — summaries
    _panel_summary(
        ax_C,
        df_summary,
        metric,
        metric_ylabel,
        "C",
        q_heal,
        p_low,
        show_legend=False,
        wu_note=wu_note,
    )
    _panel_summary(
        ax_D,
        df_summary,
        metric,
        metric_ylabel,
        "D",
        q_heal,
        p_high,
        show_legend=True,
        wu_note=wu_note,
    )

    # Row 3 — heatmaps
    _panel_heatmap(
        ax_E,
        df_summary,
        metric,
        cbar_label,
        "E",
        p_low,
        vmin,
        vmax,
        threshold_log,
        show_ylabel=True,
        show_colorbar=True,
        wu_note=wu_note,
    )
    _panel_heatmap(
        ax_F,
        df_summary,
        metric,
        cbar_label,
        "F",
        p_high,
        vmin,
        vmax,
        threshold_log,
        show_ylabel=False,
        show_colorbar=True,
        wu_note=wu_note,
    )

    if adjust_y_limits:
        for ax_left, ax_right in [(ax_A, ax_B), (ax_C, ax_D)]:
            ymin = min(ax_left.get_ylim()[0], ax_right.get_ylim()[0])
            ymax = max(ax_left.get_ylim()[1], ax_right.get_ylim()[1])
            ax_left.set_ylim(ymin, ymax)
            ax_right.set_ylim(ymin, ymax)

    return fig


# Entry point

if __name__ == "__main__":
    raw_data_path = pathlib.Path(__file__).parent.parent / "data" / "raw"
    csv_paths = raw_data_path.glob("**/main.csv")
    df = pd.concat(pd.read_csv(p) for p in csv_paths if p.stat().st_size > 0)

    shared_kwargs = dict(
        q_heal=[0.0, 0.01, 0.02, 0.05, 0.1],
        p_high=0.05,
        warm_up=100,
        adjust_y_limits=True,
        trajectory_subsets=[
            (0.0, 1.35),
            (0.0, 1.05),
            (0.01, 1.35),
            (0.01, 1.05),
            (0.05, 1.35),
            (0.05, 1.05),
        ],
    )

    fig_main = make_figure_2col(
        df,
        metric="mean_aces",
        metric_ylabel="ACEs per year",
        cbar_label="Mean ACEs",
        **shared_kwargs,
    )
    out_main = pathlib.Path(__file__).parent / "main.pdf"
    fig_main.savefig(out_main, bbox_inches="tight")
    print(f"Saved {out_main}")

    fig_si = make_figure_2col(
        df,
        metric="prop_traumatised",
        metric_ylabel="Proportion traumatised",
        cbar_label="Prop. traumatised",
        **shared_kwargs,
    )
    out_si = pathlib.Path(__file__).parent / "si.pdf"
    fig_si.savefig(out_si, bbox_inches="tight")
    print(f"Saved {out_si}")

    fig_threshold = make_threshold_vs_p_figure(
        df,
        metric="mean_aces",
        warm_up=100,
        p_high=shared_kwargs["p_high"],
    )
    out_threshold = pathlib.Path(__file__).parent / "si_threshold.pdf"
    fig_threshold.savefig(out_threshold, bbox_inches="tight")
    print(f"Saved {out_threshold}")
