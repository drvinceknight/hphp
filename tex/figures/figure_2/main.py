from __future__ import annotations

import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from hphp.analytical import (
    simulation,
    x_star_equilibrium,
    delta,
    find_roots_on_unit_interval,
    is_stable_equilibrium,
)


mpl.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica Neue", "Helvetica", "Arial"],
    "font.size":         7,
    "axes.titlesize":    7,
    "axes.labelsize":    7,
    "xtick.labelsize":   6,
    "ytick.labelsize":   6,
    "legend.fontsize":   6,
    "axes.linewidth":    0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size":  2.5,
    "ytick.major.size":  2.5,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "lines.linewidth":   1.0,
    "pdf.fonttype":      42,
    "svg.fonttype":      "none",
    "figure.dpi":        300,
})


# Colour palette (Wong, colourblind-safe)
PALETTE    = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
              "#56B4E9", "#D55E00", "#F0E442", "#000000"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "ace_heat", ["#FFFFFF", "#A8D8D8", "#2A9D8F", "#264653"], N=256
)


# Plotting utilities

def nature_axes_style(ax):
    ax.spines[["top", "right"]].set_visible(False)


def mark_equilibrium_stability(ax, f, r, ms=5.5):
    """Filled circle = stable, open circle = unstable."""
    if is_stable_equilibrium(f, r):
        ax.plot(r, 0, "o", color="black", markersize=ms)
    else:
        ax.plot(r, 0, "o", markerfacecolor="white", markeredgecolor="black", markersize=ms)


def panel_label(ax, text, dy=0.02):
    ax.text(
        0.0, 1.0 + dy, text,
        transform=ax.transAxes,
        fontsize=7, va="top", ha="left",
    )


# Panel A: trajectories

def plot_panel_A(fig, subspec, *, alpha, T, param_sets, x0_high=0.95, x0_low=0.05):
    axes = fig.add_subfigure(subspec).subplots(2, 3, sharex=True, sharey=True)
    t = np.arange(T + 1)

    eq_style = dict(color="black", linewidth=1.0, linestyle=":")

    for i, (ax, (p, q, title)) in enumerate(zip(axes.flat, param_sets)):
        col = PALETTE[i % len(PALETTE)]
        high = list(simulation(x_T=x0_high, alpha=alpha, p=p, q=q, number_of_iterations=T))
        low  = list(simulation(x_T=x0_low,  alpha=alpha, p=p, q=q, number_of_iterations=T))

        ax.plot(t[:len(high)], high, color=col, linewidth=1.8, linestyle="-")
        ax.plot(t[:len(low)],  low,  color=col, linewidth=1.6, linestyle="--")
        ax.axhline(x_star_equilibrium(alpha=alpha, p=p, q=q), **eq_style)

        ax.text(0.14, 0.96, rf"$p={p:.2f},\ q={q:.2f}$",
                transform=ax.transAxes, va="top", ha="left", fontsize=6)
        ax.set_title(title, fontsize=6)
        ax.set_xlim(0, T)
        ax.set_ylim(0, 1)
        nature_axes_style(ax)

    sf = axes.flat[0].get_figure()
    sf.supxlabel(r"Generation $t$", y=0.02, fontsize=7)
    sf.supylabel(r"Trauma prevalence $x_T(t)$", x=0.02, fontsize=7)

    sf.legend(
        [plt.Line2D([], [], color="black", linewidth=1.8, linestyle="-"),
         plt.Line2D([], [], color="black", linewidth=1.6, linestyle="--"),
         plt.Line2D([], [], **eq_style)],
        [r"high $x_T(0)$", r"low $x_T(0)$", r"equilibrium $x^*$"],
        frameon=False, loc="upper center", ncol=3, fontsize=6,
        bbox_to_anchor=(0.5, 1.08),
    )

    panel_label(axes[0, 0], r"$\mathbf{A}$  Trajectories across exposure/healing regimes", dy=0.3)
    return axes


# Panel B: Δ(x) overlay + phase arrows

def plot_panel_B(fig, subspec, *, p, q, alphas, alpha_ref):
    subfig = fig.add_subfigure(subspec)
    ax1, ax2 = subfig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [1.1, 1.0]})

    x = np.linspace(0, 1, 800)

    ref_idx = int(np.argmin(np.abs(np.asarray(alphas) - alpha_ref)))
    _other_styles = ["--", "-.", ":", (0, (3, 1, 1, 1))]
    other_count = 0
    for i, a in enumerate(alphas):
        ls = "-" if i == ref_idx else _other_styles[other_count % len(_other_styles)]
        if i != ref_idx:
            other_count += 1
        f = lambda xx, aa=a: delta(xx, aa, p, q)
        ax1.plot(x, delta(x, a, p, q), color=PALETTE[i % len(PALETTE)],
                 linestyle=ls, linewidth=1.7,
                 label=rf"$\alpha={a:.2f}$")
        for r in find_roots_on_unit_interval(f, x):
            mark_equilibrium_stability(ax1, f, r, ms=4.5)

    ax1.axhline(0, linestyle=":", color="black", linewidth=1.0)
    ax1.legend(frameon=False, loc="best", fontsize=6)
    nature_axes_style(ax1)

    ref_color = PALETTE[ref_idx % len(PALETTE)]

    y_ref = delta(x, alpha_ref, p, q)
    ax2.plot(x, y_ref, color=ref_color, linewidth=1.9)
    ax2.axhline(0, linestyle=":", color="black", linewidth=1.0)

    f_ref = lambda xx: delta(xx, alpha_ref, p, q)
    for r in find_roots_on_unit_interval(f_ref, x):
        mark_equilibrium_stability(ax2, f_ref, r, ms=4.5)

    arrow_x = np.linspace(0.06, 0.94, 12)
    y_arrow = -0.02 * (np.max(np.abs(y_ref)) + 1e-12)
    dx = 0.035
    for xx in arrow_x:
        s = np.sign(f_ref(xx))
        if s == 0:
            continue
        ax2.annotate("", xy=(xx + s * dx, y_arrow), xytext=(xx, y_arrow),
                     arrowprops=dict(arrowstyle="->", color="black", linewidth=1.0))

    ax2.set_xlabel(r"Trauma prevalence $x_T$", fontsize=7)
    nature_axes_style(ax2)

    panel_label(ax1, rf"$\mathbf{{B}}$  Map diagnostics and stability  $p={p:.2f},\ q={q:.2f}$", dy=0.2)
    subfig.supylabel(r"Net growth $\Delta(x_T) = x'_T - x_T$", x=0.02, fontsize=7)
    return ax1, ax2


# Heatmap panel

def plot_heatmap_panel(fig, subspec, label, p=0.0, *,
                       alpha_min=1.0, alpha_max=2.5, q_min=0.0, q_max=1.0, n=400,
                       show_ylabel=True):
    ax = fig.add_subplot(subspec)

    alpha_vals = np.linspace(alpha_min, alpha_max, n)
    q_vals     = np.linspace(q_min, q_max, n)
    A, Q = np.meshgrid(alpha_vals, q_vals)

    if p == 0.0:
        X_star = np.zeros_like(A)
        mask = (A > 1) & (A * (1 - Q) > 1)
        X_star[mask] = np.clip(1 - (A[mask] * Q[mask]) / (A[mask] - 1), 0, 1)
    else:
        X_star = np.zeros_like(A)
        mask = A > 1
        B    = 1 + p - A[mask] + A[mask] * Q[mask]
        disc = B**2 + 4 * p * (A[mask] - 1)
        X_star[mask] = np.clip((-B + np.sqrt(np.maximum(disc, 0))) / (2 * (A[mask] - 1)), 0, 1)

    im = ax.imshow(X_star, origin="lower", aspect="auto", cmap=HEATMAP_CMAP,
                   extent=[alpha_vals.min(), alpha_vals.max(), q_vals.min(), q_vals.max()],
                   vmin=0, vmax=1)

    alpha_curve = np.linspace(max(alpha_min, 1.001), alpha_max, 600)
    ax.plot(alpha_curve, 1 - 1 / alpha_curve, color="black", linewidth=1.8,
            linestyle="--", label=r"$q = 1 - 1/\alpha$  $(p=0)$")

    ax.set_xlabel(r"Relative reproductive multiplier $\alpha$", fontsize=7)
    if show_ylabel:
        ax.set_ylabel(r"Healing probability $q$", fontsize=7)
    nature_axes_style(ax)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"Endemic trauma level $x^*$", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    title = (rf"$\mathbf{{{label}}}$  Endemic level  $(p = {p})$")
    panel_label(ax, title, dy=0.05)
    return ax


# Figure assembly

def make_figure():
    fig = plt.figure(figsize=(14.0, 7.2))
    gs = fig.add_gridspec(
        nrows=2, ncols=3,
        height_ratios=[1.25, 1.0],
        width_ratios=[1.0, 1.0, 1.0],
        hspace=0.35, wspace=0.35,
    )

    alpha, T = 1.10, 60
    param_sets = [
        (0.00, 0.00, "no exposure, no healing"),
        (0.00, 0.20, "no exposure, strong healing"),
        (0.02, 0.02, "low exposure, low healing"),
        (0.05, 0.07, "mod. exposure, mod. healing"),
        (0.02, 0.18, "low exposure, high healing"),
        (0.12, 0.18, "high exposure, high healing"),
    ]

    plot_panel_A(fig, gs[0, 0], alpha=alpha, T=T, param_sets=param_sets)
    plot_panel_B(fig, gs[1, 0], p=0.05, q=0.07, alphas=np.linspace(1.0, 1.2, 5), alpha_ref=1.10)
    plot_heatmap_panel(fig, gs[:, 1], label="C", p=0.0, show_ylabel=True)
    plot_heatmap_panel(fig, gs[:, 2], label="D", p=0.05, show_ylabel=False)

    return fig


if __name__ == "__main__":
    fig = make_figure()
    plt.savefig(pathlib.Path(__file__).parent / "main.pdf")
