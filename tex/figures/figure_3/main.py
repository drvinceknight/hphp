"""
Specification of the simulation model.

Three mechanisms determine how adversity moves through the population: the
effect of adversity on fertility, the change in adversity within a lifetime,
and its transmission between generations. Each is shown here directly.
"""

from __future__ import annotations

import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import hphp.birth_death as birth_death

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

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]
EXPOSURE_COLOUR = "#E69F00"
HEALING_COLOUR = "#0072B2"

TRANSMISSION_CMAP = LinearSegmentedColormap.from_list(
    "transmission", ["#FFFFFF", "#A8D8D8", "#2A9D8F", "#264653"], N=256
)

AGES = np.arange(0, 61)
SHOWN_ACE_COUNTS = (0, 2, 4, 8)
ACE_COUNTS = np.arange(9)


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)


def panel_label(ax, letter, title):
    ax.set_title(f"$\\mathbf{{{letter}}}$  {title}", loc="left", fontsize=7)


def schedule_summary(number_of_aces, survivorship):
    """Net reproduction and mean age at childbearing, weighted by survival."""
    probabilities = np.array(
        [
            birth_death.fertility(
                age=age, completed_number_of_aces=number_of_aces
            )
            for age in range(101)
        ]
    )
    weighted = probabilities * survivorship[:101]
    total = weighted.sum()
    return total, (weighted * np.arange(101)).sum() / total


def draw_fertility_schedule(ax):
    for index, number_of_aces in enumerate(SHOWN_ACE_COUNTS):
        ax.plot(
            AGES,
            [
                birth_death.fertility(
                    age=age, completed_number_of_aces=number_of_aces
                )
                for age in AGES
            ],
            color=PALETTE[index],
            linestyle=["-", "--", "-.", ":"][index],
            label=f"$\\bar{{k}}$ = {number_of_aces}",
        )
    ax.axvline(
        birth_death.YOUNGEST_FERTILE_AGE, color="0.55", linewidth=0.6, linestyle=":"
    )
    ax.set_xlabel("age (years)")
    ax.set_ylabel("annual birth probability")
    ax.set_xlim(0, 60)
    ax.legend(frameon=False, handlelength=1.6, title="ACEs by age 18", title_fontsize=6)
    panel_label(ax, "A", "tempo: adversity accelerates the schedule")
    style_axes(ax)


def draw_timing_summary(ax, survivorship):
    """
    Both quantities relative to their unexposed value, so that a change in one
    can be compared with a change in the other. Separate axes would let the
    scaling decide which curve looks like it moves.
    """
    summaries = [schedule_summary(k, survivorship) for k in ACE_COUNTS]
    mean_ages = np.array([mean_age for _, mean_age in summaries])
    totals = np.array([total for total, _ in summaries])

    ax.plot(
        ACE_COUNTS,
        mean_ages / mean_ages[0],
        color=PALETTE[0],
        marker="o",
        markersize=3,
        label="mean age at childbearing",
    )
    ax.plot(
        ACE_COUNTS,
        totals / totals[0],
        color=PALETTE[2],
        marker="s",
        markersize=3,
        linestyle="--",
        label="net reproduction",
    )
    ax.axhline(1.0, color="0.75", linewidth=0.5, linestyle=":")

    for index, label in ((0, f"{mean_ages[0]:.1f} yr"), (8, f"{mean_ages[8]:.1f} yr")):
        ax.annotate(
            label,
            xy=(index, mean_ages[index] / mean_ages[0]),
            xytext=(4 if index == 0 else -4, -8),
            textcoords="offset points",
            ha="left" if index == 0 else "right",
            fontsize=5.5,
            color=PALETTE[0],
        )

    ax.set_xlabel("ACEs by age 18, $\\bar{k}$")
    ax.set_ylabel("relative to $\\bar{k} = 0$")
    ax.set_ylim(0.6, 1.15)
    ax.legend(frameon=False, handlelength=1.6, loc="lower left")
    panel_label(ax, "B", "timing moves by a third, fertility by 2%")
    style_axes(ax)


def draw_normalisation(ax, alpha=1.35):
    exposed_share = np.linspace(0, 1, 200)
    mean_multiplier = (1 - exposed_share) + exposed_share * alpha
    ax.plot(
        exposed_share,
        alpha / mean_multiplier,
        color=PALETTE[3],
        label="exposed ($k \\geq 1$)",
    )
    ax.plot(
        exposed_share,
        1 / mean_multiplier,
        color=PALETTE[0],
        linestyle="--",
        label="unexposed ($k = 0$)",
    )
    ax.plot(
        exposed_share,
        np.ones_like(exposed_share),
        color="0.55",
        linestyle=":",
        linewidth=0.8,
        label="population total",
    )
    ax.set_xlabel("share of the population exposed")
    ax.set_ylabel("fertility relative to $\\alpha = 1$")
    ax.legend(frameon=False, handlelength=1.6)
    panel_label(ax, "C", f"quantum: $\\alpha$ = {alpha} redistributes births")
    style_axes(ax)


def draw_empirical_inputs(ax, survivorship):
    """The two schedules the model is built on, both from UN data."""
    ages = np.array(sorted(birth_death.BASELINE_BIRTH_PROBABILITY_BY_AGE))
    ax.plot(
        ages,
        [birth_death.BASELINE_BIRTH_PROBABILITY_BY_AGE[age] for age in ages],
        color=EXPOSURE_COLOUR,
        label="fertility $b_0(a)$",
    )
    ax.set_xlabel("age (years)")
    ax.set_ylabel("annual birth probability", color=EXPOSURE_COLOUR)
    ax.tick_params(axis="y", labelcolor=EXPOSURE_COLOUR)
    ax.set_xlim(0, 100)
    panel_label(ax, "D", "the empirical schedules")
    style_axes(ax)

    twin = ax.twinx()
    twin.plot(
        np.arange(101),
        survivorship[:101],
        color=HEALING_COLOUR,
        linestyle="--",
        label="survivorship $l(a)$",
    )
    twin.set_ylabel("survivorship", color=HEALING_COLOUR, fontsize=6)
    twin.tick_params(axis="y", labelcolor=HEALING_COLOUR, labelsize=5)
    twin.set_ylim(0, 1.02)
    twin.spines[["top"]].set_visible(False)

    handles = ax.get_lines() + twin.get_lines()
    ax.legend(
        handles,
        [line.get_label() for line in handles],
        frameon=False,
        handlelength=1.6,
        loc="center right",
    )


def draw_within_life_probabilities(ax, trauma_probability=0.05, heal_probability=0.05):
    ages = np.arange(0, 61)
    ax.plot(
        ages,
        np.where(ages < 18, trauma_probability, 0.0),
        color=EXPOSURE_COLOUR,
        label="gain an ACE ($p$)",
    )
    ax.plot(
        ages,
        np.where(ages < 18, 0.0, heal_probability),
        color=HEALING_COLOUR,
        linestyle="--",
        label="lose an ACE ($q$)",
    )
    ax.axvline(18, color="0.3", linewidth=0.6, linestyle=":")
    ax.text(
        18,
        -0.004,
        "18",
        ha="center",
        va="top",
        fontsize=5.5,
        color="0.3",
    )
    ax.set_xlabel("age (years)")
    ax.set_ylabel("annual probability")
    ax.set_ylim(-0.005, max(trauma_probability, heal_probability) * 1.5)
    ax.legend(frameon=False, handlelength=1.6, loc="upper left")
    panel_label(
        ax,
        "E",
        f"within life: disjoint phases, $p$ = {trauma_probability}, "
        f"$q$ = {heal_probability}",
    )
    style_axes(ax)


def transmission_matrix(samples=40_000):
    """
    Pr(child ACE count | maternal ACE count), sampled from the model itself so
    that the panel cannot drift from the implementation.
    """
    generator = np.random.default_rng(0)
    matrix = np.zeros((9, 9))
    for maternal in ACE_COUNTS:
        counts = np.bincount(
            [
                birth_death.sample_intergenerational_number_of_aces(
                    number_of_maternal_aces=maternal, generator=generator
                )
                for _ in range(samples)
            ],
            minlength=9,
        )
        matrix[maternal] = counts / counts.sum()
    return matrix


def draw_transmission(ax, matrix):
    image = ax.imshow(
        matrix.T, origin="lower", cmap=TRANSMISSION_CMAP, aspect="auto", vmin=0
    )
    ax.set_xlabel("maternal ACE count")
    ax.set_ylabel("child ACE count")
    ax.set_xticks(ACE_COUNTS)
    ax.set_yticks(ACE_COUNTS)
    colourbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colourbar.set_label("probability", fontsize=6)
    colourbar.ax.tick_params(labelsize=5, width=0.5)
    colourbar.outline.set_linewidth(0.5)
    panel_label(ax, "F", "transmission between generations")


def main():
    survivorship = np.array(birth_death.survivorship("Female"))

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.0))

    draw_fertility_schedule(axes[0][0])
    draw_timing_summary(axes[0][1], survivorship)
    draw_normalisation(axes[0][2])

    draw_empirical_inputs(axes[1][0], survivorship)
    draw_within_life_probabilities(axes[1][1])
    draw_transmission(axes[1][2], transmission_matrix())

    fig.tight_layout(pad=1.2, w_pad=2.4, h_pad=2.4)
    output = pathlib.Path(__file__).parent / "main.pdf"
    fig.savefig(output, bbox_inches="tight")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
