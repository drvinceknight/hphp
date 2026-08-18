"""
Write the shell commands for the parameter sweep.

The grid is written out explicitly rather than with `numpy.linspace`, because
the output directory name is built from the formatted parameter values: a grid
whose spacing does not survive rounding to two decimal places produces
colliding directory names, and runs then silently overwrite one another.

Completed combinations are recorded by `stet`, so the full list can be piped to
a runner repeatedly and only the outstanding work is done. Delete the store
whenever the model or an unswept setting changes.
"""

import itertools

REPETITIONS = 50
YEARS = 200
INITIAL_POPULATION_SIZE = 10_000

# Healing probability: 0.00 to 0.60 in steps of 0.01.
HEAL_PROBABILITIES = [round(0.01 * i, 2) for i in range(61)]

# Reproductive multiplier: 1.05 to 1.50 in steps of 0.05.
ALPHAS = [round(1.05 + 0.05 * i, 2) for i in range(10)]

# Background exposure: the dense range plus the saturated case kept for the
# supplementary material.
TRAUMA_PROBABILITIES = [round(0.01 * i, 2) for i in range(11)] + [0.20]


def main():
    combinations = list(
        itertools.product(TRAUMA_PROBABILITIES, HEAL_PROBABILITIES, ALPHAS)
    )

    formatted = {
        f"{trauma:.2f}_{heal:.2f}_{alpha:.2f}"
        for trauma, heal, alpha in combinations
    }
    if len(formatted) != len(combinations):
        raise ValueError(
            "Grid values collide once formatted to two decimal places, which "
            "would make runs overwrite each other."
        )

    for trauma, heal, alpha in combinations:
        print(
            f"uv run main.py"
            f" --repetitions {REPETITIONS}"
            f" --years {YEARS}"
            f" --initial_population_size {INITIAL_POPULATION_SIZE}"
            f" --probability_of_trauma {trauma:.02f}"
            f" --probability_of_heal {heal:.02f}"
            f" --alpha {alpha:.02f}"
        )


if __name__ == "__main__":
    main()
