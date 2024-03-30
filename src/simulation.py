import numpy as np
import copy

from .individual import Individual
from .birth_death import (
    sample_number_of_aces,
    sample_intergenerational_number_of_aces,
    death,
    birth,
    adjust_aces,
)


def simulate(
    number_of_years,
    initial_population,
    probability_of_male_birth,
    alpha=3,
    seed=None,
    probability_of_heal=0,
    probability_of_trauma=0,
):
    """
    Simulate a population of individuals through which intergenerational trauma
    is passed on.

    Parameters
    ----------
    number_of_years: int
        Number of years to simulate
    initial_population: iterable
        A collection of Individuals
    probability_of_male_birth: float
        The probability of any given birth being Male (this is usually more than
        1/2)
    alpha: int
        An adjustment passed to the probability of giving birth based on the
        number of aces of the mother.
    seed: int
        The random seed.
    probability_of_heal: float
        The probability of decreasing the number of aces by one.
    probability_of_external_childhood_trauma: float
        The probability of a new traumatic event in childhood
    """
    np.random.seed(seed)
    populations = [initial_population]
    for _ in range(number_of_years):

        population = []

        for individual in populations[-1]:
            if (
                individual.sex == "Female"
                and birth(
                    age=individual.age,
                    number_of_aces=individual.number_of_aces,
                    alpha=alpha,
                )
                is True
            ):
                sex = (
                    "Male"
                    if np.random.random() < probability_of_male_birth
                    else "Female"
                )
                number_of_aces = sample_intergenerational_number_of_aces(
                    number_of_maternal_aces=individual.number_of_aces
                )
                population.append(
                    Individual(sex=sex, number_of_aces=number_of_aces, age=0)
                )

            if death(age=individual.age, sex=individual.sex) is False:
                new_age = individual.age + 1
                aces_delta = adjust_aces(
                    individual,
                    probability_of_heal=probability_of_heal,
                    probability_of_trauma=probability_of_trauma,
                )
                new_number_of_aces = individual.number_of_aces + aces_delta
                sex = individual.sex
                population.append(
                    Individual(age=new_age, sex=sex, number_of_aces=new_number_of_aces)
                )

        populations.append(population)

    return populations


def get_population(
    number_of_individuals,
    population_pyramid,
    seed=None,
    **population_pyramid_kwargs,
):
    """
    Create a population of individuals .

    - Number of individuals,
    - group: "Total", "Male" or "Female"
    - age_distribution: a callable that returns an age
    - age_distribution_kwargs: kwargs for age_distribution
    """
    np.random.seed(seed)
    individuals = []
    for _ in range(number_of_individuals):
        age, sex = population_pyramid(**population_pyramid_kwargs)
        individuals.append(
            Individual(
                age=age,
                sex=sex,
                number_of_aces=sample_number_of_aces(sex=sex),
            )
        )
    return individuals


def uk_population_pyramid():
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
    age_group_index = np.random.choice(
        a=range(len(age_groups)),
        p=age_group_probabilities,
    )

    lower_bound, upper_bound = age_groups[age_group_index]
    probability_of_male = probability_of_male_by_age_group[age_group_index]
    sex = "Male" if np.random.random() < probability_of_male else "Female"

    return np.random.randint(lower_bound, upper_bound + 1), sex
