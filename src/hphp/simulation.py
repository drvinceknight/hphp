from collections.abc import Callable
from typing import Any

import numpy as np

from . import birth_death
from .birth_death import (
    adjust_aces,
    birth,
    death,
    sample_intergenerational_number_of_aces,
    sample_number_of_aces,
)
from .individual import Individual


def mean_fertility_multiplier(
    population: list[Individual],
    alpha: float,
    pace_increase_per_ace: float,
) -> float:
    """
    Population mean of the quantum multiplier, weighted by baseline fertility.

    The analytical model is a replicator dynamic, in which dividing by mean
    fitness normalises population size away: alpha there is a purely relative
    advantage that redistributes reproductive share between types. Without the
    same normalisation the simulation would treat alpha as an absolute increase
    in fertility, so that raising alpha inflates the whole population rather
    than shifting births towards the exposed. Dividing every individual's
    multiplier by this mean holds total expected births at their alpha = 1
    level while preserving the ratio between exposed and unexposed.
    """
    weighted_total = 0.0
    baseline_total = 0.0
    for individual in population:
        if individual.sex != "Female":
            continue
        baseline = birth_death.fertility(
            age=individual.age,
            completed_number_of_aces=individual.completed_number_of_aces,
            pace_increase_per_ace=pace_increase_per_ace,
        )
        if baseline == 0.0:
            continue
        multiplier = alpha if individual.number_of_aces >= 1 else 1.0
        weighted_total += baseline * multiplier
        baseline_total += baseline
    if baseline_total == 0.0:
        return 1.0
    return weighted_total / baseline_total


def simulate(
    number_of_years: int,
    initial_population: list[Individual],
    probability_of_male_birth: float,
    alpha: float = 1.1,
    pace_increase_per_ace: float = 0.3142,
    seed: int | None = None,
    generator: np.random.Generator | None = None,
    normalise_fertility: bool = True,
    probability_of_heal: float = 0,
    probability_of_trauma: float = 0,
) -> list[list[Individual]]:
    """
    Simulate an age-structured population in which adverse childhood
    experiences (ACEs) influence fertility through both timing (tempo)
    and intensity (quantum) effects.

    The model separates two mechanisms:

        1. Tempo effect:
           Individuals with higher ACE exposure reproduce earlier.
           This is implemented by rescaling the fertility schedule about
           the youngest fertile age by a pace of
           1 + `pace_increase_per_ace` per ACE, at unchanged completed
           fertility.

        2. Quantum effect:
           Conditional on age, individuals with ACE exposure have a
           multiplicative increase in fertility intensity controlled by
           `alpha`. Treating ACE=0 as unexposed and ACE>=1 as exposed:
           f_T = alpha * f_S.

    Parameters
    ----------
    number_of_years : int
        Number of years to simulate.

    initial_population : iterable of Individual
        Initial population state.

    probability_of_male_birth : float
        Probability that a newborn is male.

    alpha : float, default=1.1
        Multiplicative fertility factor (quantum effect). Values greater
        than 1 imply higher reproductive intensity for ACE-exposed
        individuals.

    pace_increase_per_ace : float, default=0.3142
        Increase in the life-history pace per ACE (tempo effect), motivated by
        empirical associations between ACE exposure and earlier age at first
        birth. The default is calibrated so one ACE lowers the mean age at
        childbearing by three years.

    seed : int or None
        Seed used to build a generator when one is not supplied.

    generator : numpy.random.Generator or None
        Random generator. Pass the same generator to `get_population` to
        continue a single stream across the initial population and the run,
        rather than restarting the same stream twice.

    normalise_fertility : bool, default=True
        Hold total expected births at their alpha = 1 level by dividing each
        individual's quantum multiplier by the population mean, so that alpha
        is a relative advantage as it is in the replicator model. Setting this
        to False recovers the unnormalised behaviour, in which raising alpha
        also raises the population growth rate.

    probability_of_heal : float
        Annual probability that an adult reduces their ACE count by one.

    probability_of_trauma : float
        Annual probability that a child experiences an additional ACE.

    Returns
    -------
    populations : list of lists
        A history of the population at each year.
    """
    if generator is None:
        generator = np.random.default_rng(seed)
    populations: list[list[Individual]] = [initial_population]
    for _ in range(number_of_years):

        population: list[Individual] = []
        normaliser = (
            mean_fertility_multiplier(
                populations[-1], alpha, pace_increase_per_ace
            )
            if normalise_fertility
            else 1.0
        )

        for individual in populations[-1]:
            if (
                individual.sex == "Female"
                and birth(
                    age=individual.age,
                    number_of_aces=individual.number_of_aces,
                    alpha=alpha,
                    pace_increase_per_ace=pace_increase_per_ace,
                    completed_number_of_aces=individual.completed_number_of_aces,
                    mean_fertility_multiplier=normaliser,
                    generator=generator,
                )
                is True
            ):
                sex = (
                    "Male"
                    if generator.random() < probability_of_male_birth
                    else "Female"
                )
                number_of_aces = sample_intergenerational_number_of_aces(
                    number_of_maternal_aces=individual.number_of_aces,
                    generator=generator,
                )
                population.append(
                    Individual(
                        sex=sex,
                        number_of_aces=number_of_aces,
                        age=0,
                        completed_number_of_aces=number_of_aces,
                    )
                )

            if (
                death(
                    age=individual.age, sex=individual.sex, generator=generator
                )
                is False
            ):
                new_age = individual.age + 1
                aces_delta = adjust_aces(
                    individual,
                    probability_of_heal=probability_of_heal,
                    probability_of_trauma=probability_of_trauma,
                    generator=generator,
                )
                new_number_of_aces = individual.number_of_aces + aces_delta
                sex = individual.sex
                # Childhood adversity is still accumulating below 18; from 18
                # onwards it is frozen, so that the life-history pace it sets
                # is not undone by later healing.
                completed_number_of_aces = (
                    new_number_of_aces
                    if new_age < 18
                    else individual.completed_number_of_aces
                )
                population.append(
                    Individual(
                        age=new_age,
                        sex=sex,
                        number_of_aces=new_number_of_aces,
                        completed_number_of_aces=completed_number_of_aces,
                    )
                )

        populations.append(population)

    return populations


def get_population(
    number_of_individuals: int,
    population_pyramid: Callable[..., tuple[int, str]],
    seed: int | None = None,
    generator: np.random.Generator | None = None,
    normalise_fertility: bool = True,
    **population_pyramid_kwargs: Any,
) -> list[Individual]:
    """
    Create a population by sampling from `population_pyramid`.

    Pass the same generator to `simulate` to keep one continuous stream across
    the initial population and the run.
    """
    if generator is None:
        generator = np.random.default_rng(seed)
    individuals: list[Individual] = []
    for _ in range(number_of_individuals):
        age, sex = population_pyramid(
            generator=generator, **population_pyramid_kwargs
        )
        number_of_aces = sample_number_of_aces(sex=sex, generator=generator)
        individuals.append(
            Individual(
                age=age,
                sex=sex,
                number_of_aces=number_of_aces,
                completed_number_of_aces=number_of_aces,
            )
        )
    return individuals


def uk_population_pyramid(
    generator: np.random.Generator | None = None,
) -> tuple[int, str]:
    """
    Based on this data from the Office of National Statistics

    0-4	1784061	1692448
    5-9	2046155	1942515
    10-14	2150166	2046675
    15-19	2020974	1921110
    20-24	1989729	1884226
    25-29	2246300	2126939
    30-34	2332064	2272413
    35-39	2221959	2255614
    40-44	2161673	2209186
    45-49	1982753	2021878
    50-54	2194108	2271137
    55-59	2251625	2343683
    60-64	2046039	2140963
    65-69	1708983	1817892
    70-74	1514497	1663233
    75-79	1337485	1527056
    80-84	795354	1000676
    85-89	464994	663263
    90-94	186735	327263
    95-99	44769	104314
    100	5159	18725
    """
    age_groups = (
        (0, 4),
        (5, 9),
        (10, 14),
        (15, 19),
        (20, 24),
        (25, 29),
        (30, 34),
        (35, 39),
        (40, 44),
        (45, 49),
        (50, 54),
        (55, 59),
        (60, 64),
        (65, 69),
        (70, 74),
        (75, 79),
        (80, 84),
        (85, 89),
        (90, 94),
        (95, 99),
        (100, 100),
    )
    age_group_probabilities = (
        0.0513237924129001,
        0.058884838521506,
        0.0619580723863934,
        0.0581970881968707,
        0.057191298005245,
        0.0645622406293206,
        0.067976013212672,
        0.0661025261737008,
        0.0645271046276757,
        0.0591204711779157,
        0.0659205275903903,
        0.0678406510281835,
        0.0618128189745511,
        0.052067346975442,
        0.046912910297153,
        0.042289292978169,
        0.0265148374094072,
        0.0166564873142573,
        0.00758816578718646,
        0.00220091619043483,
        0.000352600110625258,
    )
    probability_of_male_by_age_group = (
        0.513176005009623,
        0.512991799271436,
        0.512329630786584,
        0.512666396758669,
        0.51361696250989,
        0.513646750154748,
        0.506477500050494,
        0.496241825649744,
        0.494564798361146,
        0.495115030573354,
        0.491374605424786,
        0.48998347880055,
        0.488664442959425,
        0.484560127591707,
        0.47659713065616,
        0.466910754637479,
        0.442840041647411,
        0.412134823892074,
        0.363299078984743,
        0.3002958083752,
        0.216002344665885,
    )
    draws = birth_death.resolve_generator(generator)
    age_group_index = draws.choice(
        a=np.arange(len(age_groups)),
        p=age_group_probabilities,
    )

    lower_bound, upper_bound = age_groups[age_group_index]
    probability_of_male = probability_of_male_by_age_group[age_group_index]
    sex = "Male" if draws.random() < probability_of_male else "Female"

    return int(draws.integers(lower_bound, upper_bound + 1)), sex
