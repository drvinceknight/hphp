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


def _kmeans_linear_fit(pivot, otsu_log):
    """
    For each alpha column, find the q (heal) value at which log1p(metric)
    crosses the k-means threshold, then fit q* = m * alpha + c by OLS.

    Returns (m, c, r2) — slope, intercept, R².
    Returns None if the contour does not cross in every column.
    """
    log_vals = np.log1p(pivot.values)
    q_vals = pivot.index.values.astype(float)  # probability_of_heal axis
    a_vals = pivot.columns.values.astype(float)

    crossing_q = []
    crossing_a = []
    for j, alpha in enumerate(a_vals):
        col = log_vals[:, j]
        for i in range(len(q_vals) - 1):
            if (col[i] - otsu_log) * (col[i + 1] - otsu_log) <= 0:
                dv = col[i + 1] - col[i]
                if abs(dv) < 1e-12:
                    q_cross = q_vals[i]
                else:
                    q_cross = q_vals[i] + (otsu_log - col[i]) / dv * (
                        q_vals[i + 1] - q_vals[i]
                    )
                crossing_q.append(q_cross)
                crossing_a.append(alpha)
                break

    if len(crossing_a) < 2:
        return None

    crossing_a = np.array(crossing_a)
    crossing_q = np.array(crossing_q)
    A = np.column_stack([crossing_a, np.ones_like(crossing_a)])
    m, c = np.linalg.lstsq(A, crossing_q, rcond=None)[0]
    ss_res = ((crossing_q - (m * crossing_a + c)) ** 2).sum()
    ss_tot = ((crossing_q - crossing_q.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    return m, c, r2


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
    otsu_log,
    show_ylabel=True,
    show_colorbar=False,
    wu_note="",
):
    """log1p heatmap of mean `column` with Otsu contour and linear-fit title."""
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
            levels=[otsu_log],
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

    fit = _kmeans_linear_fit(pivot, otsu_log)
    if fit is not None:
        m, c, r2 = fit
        sign = "+" if c >= 0 else "-"
        fit_str = (
            f"threshold: q* = {m:.3f}\u03b1 {sign} {abs(c):.3f}  " f"($R^2$={r2:.2f})"
        )
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
        cb.ax.axhline(otsu_log, color="#E63946", linewidth=1.2, linestyle="--")
        cb.ax.text(
            1.6,
            otsu_log,
            f"{np.expm1(otsu_log):.2g}",
            transform=cb.ax.get_yaxis_transform(),
            va="center",
            ha="left",
            fontsize=5,
            color="#E63946",
        )


def make_threshold_vs_p_figure(df, metric, warm_up=0, figsize=(7.0, 4.5)):
    """
    For every value of p in the data, compute the per-p k-means threshold and
    find the resulting contour in (q, alpha) space.  All contours are overlaid
    in a single panel, coloured by p, so the shift of the threshold with
    increasing background exposure is visible directly.
    """
    df_summary = df[df["year"] >= warm_up]
    p_vals = sorted(df_summary["probability_of_trauma"].unique())

    fig, ax = plt.subplots(figsize=figsize)

    for i, p in enumerate(p_vals):
        sub = df_summary[df_summary["probability_of_trauma"] == p]
        pivot = sub.groupby(["probability_of_heal", "alpha"])[metric].mean().unstack()
        pivot = pivot.dropna(axis=1)
        if pivot.shape[0] < 2 or pivot.shape[1] < 2:
            continue

        log_v = np.log1p(pivot.values)
        flat = log_v.flatten()
        flat = flat[np.isfinite(flat)]
        if len(flat) < 4:
            continue

        km = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=0)
        km.fit(flat[:, None])
        c0, c1 = sorted(km.cluster_centers_.flatten())
        threshold = (c0 + c1) / 2

        q_arr = pivot.index.values.astype(float)
        a_arr = pivot.columns.values.astype(float)
        crossing_a, crossing_q = [], []
        for j, alpha in enumerate(a_arr):
            col = log_v[:, j]
            for k in range(len(q_arr) - 1):
                if (col[k] - threshold) * (col[k + 1] - threshold) <= 0:
                    dv = col[k + 1] - col[k]
                    if abs(dv) < 1e-12:
                        q_cross = q_arr[k]
                    else:
                        q_cross = q_arr[k] + (threshold - col[k]) / dv * (
                            q_arr[k + 1] - q_arr[k]
                        )
                    crossing_a.append(alpha)
                    crossing_q.append(q_cross)
                    break

        if len(crossing_a) < 2:
            continue

        col = PALETTE[i % len(PALETTE)]
        ls = LINESTYLES[i % len(LINESTYLES)]
        mk = MARKERS[i % len(MARKERS)]
        n_pts = len(crossing_a)
        mark_every = max(1, n_pts // min(5, n_pts))
        ax.plot(
            crossing_a,
            crossing_q,
            color=col,
            linestyle=ls,
            marker=mk,
            markersize=3,
            markevery=mark_every,
            linewidth=1.0,
            label=f"p = {p:.2g}",
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
        "Interpolated k-means threshold in (\u03b1, q) space across all values of p",
        loc="left",
        fontsize=7,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
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
    otsu = _kmeans_threshold(all_log)

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
        otsu,
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
        otsu,
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
    raw_data_path = pathlib.Path(__file__).parent / "data" / "raw"
    csv_paths = raw_data_path.glob("**/main.csv")
    df = pd.concat(pd.read_csv(p) for p in csv_paths if p.stat().st_size > 0)

    shared_kwargs = dict(
        q_heal=[0.0, 0.01, 0.02, 0.05, 0.1],
        p_high=0.2,
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
    )
    out_threshold = pathlib.Path(__file__).parent / "si_threshold.pdf"
    fig_threshold.savefig(out_threshold, bbox_inches="tight")
    print(f"Saved {out_threshold}")
