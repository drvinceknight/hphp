import argparse
import os
import csv
import numpy as np

import hphp.simulation as sim

PYRAMID_YEARS = [0, 50, 100, 150, 200]
AGE_BIN_SIZE = 1
AGE_BINS = list(range(0, 101))  # [0, 1, 2, ..., 100]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run hphp simulations and write results to CSV."
    )

    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--years", type=int, default=100)
    parser.add_argument("--initial_population_size", type=int, default=1000)

    parser.add_argument("--alpha", type=float, default=1.1)

    parser.add_argument("--probability_of_male_birth", type=float, default=0.51)
    parser.add_argument("--probability_of_heal", type=float, default=0.01)
    parser.add_argument("--probability_of_trauma", type=float, default=0.01)

    parser.add_argument(
        "--trauma_threshold",
        type=int,
        default=1,
        help="Traumatised iff number_of_aces >= trauma_threshold.",
    )

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="raw")

    return parser.parse_args()


def make_output_directory(args):
    dir_name = (
        f"alpha{args.alpha}_"
        f"male{args.probability_of_male_birth}_"
        f"heal{args.probability_of_heal}_"
        f"trauma{args.probability_of_trauma}_"
        f"thr{args.trauma_threshold}_"
        f"reps{args.repetitions}"
    )

    full_path = os.path.join(args.output_dir, dir_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path


def summarise_population(individuals, trauma_threshold: int):
    """Return simple summary statistics for a population."""
    size = len(individuals)
    if size == 0:
        return {
            "population_size": 0,
            "males": 0,
            "females": 0,
            "mean_age": np.nan,
            "mean_aces": np.nan,
            "prop_traumatised": np.nan,
            "mean_age_traumatised": np.nan,
            "mean_age_not_traumatised": np.nan,
        }

    males = sum(1 for ind in individuals if ind.sex == "Male")
    females = size - males

    ages = np.array([ind.age for ind in individuals], dtype=float)
    aces = np.array([ind.number_of_aces for ind in individuals], dtype=float)

    traum = aces >= trauma_threshold
    n_tr = int(np.sum(traum))

    prop_tr = n_tr / size

    # Means by trauma state; NaN if group is empty
    mean_age_tr = float(np.mean(ages[traum])) if n_tr > 0 else np.nan
    mean_age_nt = float(np.mean(ages[~traum])) if n_tr < size else np.nan

    return {
        "population_size": size,
        "males": males,
        "females": females,
        "mean_age": float(np.mean(ages)),
        "mean_aces": float(np.mean(aces)),
        "prop_traumatised": float(prop_tr),
        "mean_age_traumatised": mean_age_tr,
        "mean_age_not_traumatised": mean_age_nt,
    }


def age_distribution(individuals):
    """Count males and females in each 5-year age bin."""
    n_bins = len(AGE_BINS)
    male_counts = np.zeros(n_bins, dtype=int)
    female_counts = np.zeros(n_bins, dtype=int)
    for ind in individuals:
        bin_idx = min(int(ind.age) // AGE_BIN_SIZE, n_bins - 1)
        if ind.sex == "Male":
            male_counts[bin_idx] += 1
        else:
            female_counts[bin_idx] += 1
    return male_counts, female_counts


def main():
    args = parse_args()

    output_path = make_output_directory(args)
    csv_path = os.path.join(output_path, "main.csv")
    pyramid_path = os.path.join(output_path, "pyramids.csv")

    # ---- collect all input parameters once ----
    input_params = {
        "repetitions": args.repetitions,
        "years": args.years,
        "initial_population_size": args.initial_population_size,
        "alpha": args.alpha,
        "probability_of_male_birth": args.probability_of_male_birth,
        "probability_of_heal": args.probability_of_heal,
        "probability_of_trauma": args.probability_of_trauma,
        "trauma_threshold": args.trauma_threshold,
        "base_seed": args.seed,
    }

    fieldnames = [
        "rep",
        "year",
        *input_params.keys(),
        "population_size",
        "males",
        "females",
        "mean_age",
        "mean_aces",
        "prop_traumatised",
        "mean_age_traumatised",
        "mean_age_not_traumatised",
    ]

    rows = []
    pyramid_rows = []
    pyramid_fieldnames = ["rep", "year", "age_group", "males", "females"]

    for rep in range(args.repetitions):

        seed = None if args.seed is None else args.seed + rep

        initial_population = sim.get_population(
            number_of_individuals=args.initial_population_size,
            population_pyramid=sim.uk_population_pyramid,
            seed=seed,
        )

        history = sim.simulate(
            number_of_years=args.years,
            initial_population=initial_population,
            probability_of_male_birth=args.probability_of_male_birth,
            alpha=args.alpha,
            probability_of_heal=args.probability_of_heal,
            probability_of_trauma=args.probability_of_trauma,
            seed=seed,
        )

        for year, population in enumerate(history):
            s = summarise_population(population, args.trauma_threshold)

            row = {
                "rep": rep,
                "year": year,
                **input_params,
                **s,
            }

            rows.append(row)

            if year in PYRAMID_YEARS:
                male_counts, female_counts = age_distribution(population)
                for i, age_group in enumerate(AGE_BINS):
                    pyramid_rows.append(
                        {
                            "rep": rep,
                            "year": year,
                            "age_group": age_group,
                            "males": int(male_counts[i]),
                            "females": int(female_counts[i]),
                        }
                    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(pyramid_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pyramid_fieldnames)
        writer.writeheader()
        writer.writerows(pyramid_rows)

    print(f"Wrote results to {csv_path}")
    print(f"Wrote pyramids to {pyramid_path}")


if __name__ == "__main__":
    main()
