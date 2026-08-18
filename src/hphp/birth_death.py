import numpy as np
import numpy.typing as npt

from .individual import Individual


def death(age: int, sex: str) -> bool:
    """
    This uses data from the UN's department of economic and social affairs
    https://population.un.org/wpp/Download/Standard/Mortality/
    returning True if the individual dies at the given age.

    Life Tables - Based on 1985.

    """
    if sex == "Male":
        probability_of_death_at_give_age = [
            0.07446,
            0.01592,
            0.00925,
            0.00641,
            0.00485,
            0.00377,
            0.00299,
            0.00241,
            0.00199,
            0.00172,
            0.00157,
            0.00150,
            0.00150,
            0.00153,
            0.00161,
            0.00173,
            0.00189,
            0.00209,
            0.00230,
            0.00248,
            0.00265,
            0.00279,
            0.00290,
            0.00303,
            0.00311,
            0.00312,
            0.00310,
            0.00309,
            0.00313,
            0.00317,
            0.00326,
            0.00334,
            0.00345,
            0.00356,
            0.00368,
            0.00385,
            0.00400,
            0.00407,
            0.00433,
            0.00455,
            0.00488,
            0.00516,
            0.00550,
            0.00596,
            0.00634,
            0.00686,
            0.00736,
            0.00785,
            0.00848,
            0.00909,
            0.00992,
            0.01064,
            0.01154,
            0.01240,
            0.01327,
            0.01428,
            0.01534,
            0.01660,
            0.01805,
            0.01963,
            0.02168,
            0.02369,
            0.02580,
            0.02821,
            0.03048,
            0.03334,
            0.03566,
            0.03874,
            0.04193,
            0.04536,
            0.04968,
            0.05367,
            0.05856,
            0.06368,
            0.06924,
            0.07541,
            0.08165,
            0.08858,
            0.09608,
            0.10417,
            0.11372,
            0.12342,
            0.13357,
            0.14442,
            0.15646,
            0.16863,
            0.17992,
            0.19350,
            0.20749,
            0.22131,
            0.23581,
            0.25014,
            0.26705,
            0.28080,
            0.29747,
            0.31033,
            0.32555,
            0.34112,
            0.35664,
            0.37224,
            1.0000,
        ]
    else:
        probability_of_death_at_give_age = [
            0.07033,
            0.01503,
            0.00883,
            0.00630,
            0.00494,
            0.00393,
            0.00312,
            0.00246,
            0.00194,
            0.00159,
            0.00141,
            0.00132,
            0.00130,
            0.00131,
            0.00137,
            0.00145,
            0.00155,
            0.00166,
            0.00177,
            0.00183,
            0.00188,
            0.00192,
            0.00197,
            0.00205,
            0.00211,
            0.00214,
            0.00216,
            0.00217,
            0.00222,
            0.00226,
            0.00231,
            0.00236,
            0.00242,
            0.00247,
            0.00254,
            0.00264,
            0.00280,
            0.00288,
            0.00306,
            0.00326,
            0.00346,
            0.00360,
            0.00376,
            0.00393,
            0.00409,
            0.00433,
            0.00459,
            0.00488,
            0.00530,
            0.00571,
            0.00622,
            0.00670,
            0.00720,
            0.00766,
            0.00811,
            0.00865,
            0.00927,
            0.01002,
            0.01096,
            0.01199,
            0.01329,
            0.01457,
            0.01605,
            0.01767,
            0.01921,
            0.02124,
            0.02314,
            0.02537,
            0.02758,
            0.02976,
            0.03265,
            0.03525,
            0.03877,
            0.04252,
            0.04665,
            0.05145,
            0.05614,
            0.06161,
            0.06740,
            0.07378,
            0.08176,
            0.08972,
            0.09839,
            0.10751,
            0.11900,
            0.13060,
            0.14029,
            0.15284,
            0.16478,
            0.17836,
            0.19303,
            0.20629,
            0.22322,
            0.23829,
            0.25346,
            0.27026,
            0.28709,
            0.30468,
            0.32225,
            0.33985,
            1.0000,
        ]
    try:
        probability_of_death = probability_of_death_at_give_age[age]
    except IndexError:
        probability_of_death = 1
    return bool(np.random.random() < probability_of_death)


BASELINE_BIRTH_PROBABILITY_BY_AGE = {
    15: 0.013355,
    16: 0.025160000000000002,
    17: 0.040733,
    18: 0.058070000000000004,
    19: 0.07627800000000001,
    20: 0.096371,
    21: 0.110794,
    22: 0.12112600000000001,
    23: 0.12814699999999998,
    24: 0.13259,
    25: 0.134492,
    26: 0.133936,
    27: 0.131562,
    28: 0.127503,
    29: 0.121737,
    30: 0.113844,
    31: 0.105234,
    32: 0.096591,
    33: 0.087627,
    34: 0.078933,
    35: 0.07048399999999999,
    36: 0.061793999999999995,
    37: 0.053277,
    38: 0.044770000000000004,
    39: 0.036818,
    40: 0.029645,
    41: 0.023043,
    42: 0.017268000000000002,
    43: 0.012609,
    44: 0.009054,
    45: 0.006445,
    46: 0.0046559999999999995,
    47: 0.003414,
    48: 0.0025670000000000003,
    49: 0.0017,
}

# The youngest age at which the UN schedule assigns non-zero fertility. The
# tempo effect rescales the age axis about this point.
YOUNGEST_FERTILE_AGE = 15

# Calibrated so that one ACE lowers the mean age at childbearing by 3.0 years,
# matching the estimate in 'Adverse Childhood Experiences, Early and Nonmarital
# Fertility, and Women's Health at Midlife'.
PACE_INCREASE_PER_ACE = 0.3142


def life_history_pace(
    number_of_aces: int, pace_increase_per_ace: float = PACE_INCREASE_PER_ACE
) -> float:
    """
    Return the pace at which an individual moves through the fertility
    schedule, relative to an unexposed individual.

    A pace greater than one is the accelerated life-history strategy associated
    with early adversity: the whole reproductive career runs faster, so
    childbearing happens earlier.
    """
    return 1.0 + pace_increase_per_ace * number_of_aces


def baseline_birth_probability(effective_age: float) -> float:
    """
    Linear interpolation of the UN age-specific fertility schedule.

    Interpolation is needed because the effective age is no longer an integer
    once the age axis is rescaled by the life-history pace.
    """
    if effective_age < YOUNGEST_FERTILE_AGE or effective_age >= 50:
        return 0.0
    lower_age = int(np.floor(effective_age))
    fraction = effective_age - lower_age
    lower = BASELINE_BIRTH_PROBABILITY_BY_AGE.get(lower_age, 0.0)
    upper = BASELINE_BIRTH_PROBABILITY_BY_AGE.get(lower_age + 1, 0.0)
    return lower * (1.0 - fraction) + upper * fraction


def adjust_age_for_aces(
    age: int,
    number_of_aces: int,
    pace_increase_per_ace: float = PACE_INCREASE_PER_ACE,
) -> float:
    """
    Return the effective age at which an individual's birth probability is
    evaluated.

    This is an approximation but is based on the first table from
    'Adverse Childhood Experiences, Early and Nonmarital Fertility,
    and Women's Health at Midlife'

    In the first table of that paper the probability of having a first birth in
    a given age group are given relative to the number of aces:

    # ACEs | 14-19 | 20-24 | 25-39 |
    --------------------------------
        0  | 44.6% | 45.9% | 56.8% |
    --------------------------------
        1  | 30.8% | 29.3% | 22.4% |
    --------------------------------
        2  | 14.0% | 13.5% | 13.0% |
    --------------------------------
        3+ | 10.6% | 11.2% | 7.7%  |
    --------------------------------

    Further to this a logistic regression is carried out offering the following
    approximation:

    > The estimated effect of each additional ACE in lowering age at first birth
    is equivalent to the effect of being born to a mother approximately three
    year younger.

    We implement this as an acceleration of the life-history schedule rather
    than as a rigid translation of it. The age axis is rescaled about the
    youngest fertile age,

        effective age = 15 + (age - 15) * pace,

    and the birth probability is scaled by the same pace (see `birth`), so that
    completed fertility is preserved. A rigid translation cannot do this: it
    slides the fertile window below the youngest fertile age, and for large
    ACE counts it
    pushes the whole window off the bottom of the schedule, which both
    destroys completed fertility and produces impossible ages at birth.

    Because the schedule is compressed rather than moved, the shift in mean age
    at childbearing saturates on its own (3.0 years at one ACE, 9.0 at eight),
    so no cap on the ACE count is needed.
    """
    if age < YOUNGEST_FERTILE_AGE:
        return float(age)
    pace = life_history_pace(number_of_aces, pace_increase_per_ace)
    return YOUNGEST_FERTILE_AGE + (age - YOUNGEST_FERTILE_AGE) * pace


def birth(
    age: int,
    number_of_aces: int = 0,
    alpha: float = 1.0,
    pace_increase_per_ace: float = PACE_INCREASE_PER_ACE,
) -> bool:
    """
    This uses data from the UN's department of economic and social affairs
    - <https://population.un.org/wpp/Download/Standard/Fertility/>
    to return a boolean representing whether or not someone (able to have a
    birth) has a birth at a given age.

    Parameters
    ----------
    age : int
        The biological age.
    number_of_aces : int
        Number of adverse childhood experiences.
    alpha : float
        Multiplicative fertility factor (replicator-style fitness multiplier).
        Under a coarse-grained two-type model (ACE=0 vs ACE>=1), this corresponds
        directly to the analytical parameter via f_T = alpha * f_S.
    pace_increase_per_ace : float
        Increase in the life-history pace per ACE (the tempo effect).

    We separate two mechanisms:

      (1) Tempo: earlier reproductive timing, at unchanged completed fertility.
      (2) Quantum: multiplicative increase in fertility intensity.
    """
    if age < YOUNGEST_FERTILE_AGE:
        return False

    # Tempo effect: compress the age axis and scale the hazard by the same
    # pace, so that completed fertility is unchanged and only its timing moves.
    pace = life_history_pace(number_of_aces, pace_increase_per_ace)
    effective_age = adjust_age_for_aces(
        age=age,
        number_of_aces=number_of_aces,
        pace_increase_per_ace=pace_increase_per_ace,
    )
    probability = pace * baseline_birth_probability(effective_age)

    # Quantum effect (replicator-style multiplier)
    # Minimal two-type interpretation: traumatised if ACE >= 1
    if number_of_aces >= 1:
        probability = alpha * probability

    return bool(np.random.random() < min(1.0, probability))


def sample_number_of_aces(sex: str) -> int:
    """
    This uses data from

    "Prevalence of adverse childhood experiences among individuals aged 45 to 85 years:
    a cross-sectional analysis of the Canadian Longitudinal Study on Aging"

    The table is from the supplementary material:

    # ACEs | Total | Male   | Female |
    ----------------------------------
        0  | 38.4  | 40.6   | 36.3   |
    ----------------------------------
        1  | 26.0  | 26.8   | 25.2   |
    ----------------------------------
        2  | 15.5  | 15.3   | 15.7   |
    ----------------------------------
        3  | 9.4   | 9.2    | 9.6    |
    ----------------------------------
        4  | 5.6   | 4.6    | 6.6    |
    ----------------------------------
        5  | 3.0   | 2.2    | 3.9    |
    ----------------------------------
        6  | 1.5   | 1.1    | 1.9    |
    ----------------------------------
        7  | 0.5   | 0.2    | 0.8    |
    ----------------------------------
        8  | 0.1   | 0.1    | 0.2    |
    ----------------------------------

    group: One of "Total", "Male" or "Female"
    """
    ace_data = {
        0: {"Total": 38.4, "Male": 40.6, "Female": 36.3},
        1: {"Total": 26.0, "Male": 26.8, "Female": 25.2},
        2: {"Total": 15.5, "Male": 15.3, "Female": 15.7},
        3: {"Total": 9.4, "Male": 9.2, "Female": 9.6},
        4: {"Total": 5.6, "Male": 4.6, "Female": 6.6},
        5: {"Total": 3.0, "Male": 2.2, "Female": 3.9},
        6: {"Total": 1.5, "Male": 1.1, "Female": 1.9},
        7: {"Total": 0.5, "Male": 0.2, "Female": 0.8},
        8: {"Total": 0.1, "Male": 0.1, "Female": 0.2},
    }
    aces_range = range(9)
    p: npt.NDArray[np.float64] = np.array(
        [ace_data[number][sex] for number in aces_range]
    )
    p = p / p.sum()
    return int(np.random.choice(a=aces_range, p=p))


def sample_intergenerational_number_of_aces(number_of_maternal_aces: int) -> int:
    """
    This samples the number of aces of a child based on the number of aces from
    a mother.

    This is based on this table from: "Intergenerational Associations between
    Parents' and Children's Adverse Childhood Experience Scores"

    Which contains this table (Table 3):

    Mother ACES: 0                1                2-3              4+
    Child 0:     42.6% (38.4-46.7) 37.8% (32.4-46.7) 30.4% (24.7-36.1) 25.1% (16.2-34.0)
    Child 1:     25.8% (21.9-29.6) 29.7% (23.9-35.5) 28.4% (22.2-34.7) 23.2% (14.1-32.2)
    Child 2-3:   25.8% (21.9-29.7) 21.5% (16.3-26.7) 21.3% (15.7-26.8) 23.8% (15.6-32.0)
    Child 4+:    5.8% (4.0-7.7)    11.0% (6.3-15.6)  19.9% (13.7-26.1) 27.9% (18.9-36.9)
    """
    p: npt.NDArray[np.float64]
    if number_of_maternal_aces == 0:
        p = np.array(
            (
                0.426,
                0.258,
                0.258 / 2,
                0.258 / 2,
                0.058 / 5,
                0.058 / 5,
                0.058 / 5,
                0.058 / 5,
                0.058 / 5,
            )
        )
    if number_of_maternal_aces == 1:
        p = np.array(
            (
                0.378,
                0.297,
                0.215 / 2,
                0.215 / 2,
                0.11 / 5,
                0.11 / 5,
                0.11 / 5,
                0.11 / 5,
                0.11 / 5,
            )
        )
    if number_of_maternal_aces in (2, 3):
        p = np.array(
            (
                0.304,
                0.284,
                0.213 / 2,
                0.213 / 2,
                0.199 / 5,
                0.199 / 5,
                0.199 / 5,
                0.199 / 5,
                0.199 / 5,
            )
        )
    if number_of_maternal_aces >= 4:
        p = np.array(
            (
                0.251,
                0.232,
                0.238 / 2,
                0.238 / 2,
                0.279 / 5,
                0.279 / 5,
                0.279 / 5,
                0.279 / 5,
                0.279 / 5,
            )
        )
    aces_range = range(9)
    p = p / p.sum()
    return int(np.random.choice(a=aces_range, p=p))


def adjust_aces(
    individual: Individual, probability_of_heal: float, probability_of_trauma: float
) -> int:
    """
    Return the delta of the number of aces

    - A child (less than 18) can gain another ace from an external factor.
    - An adult (18 or more) can reduce their number of aces by 1.

    The two cases are mutually exclusive: healing represents adult recovery
    processes, so a child whose trauma draw fails does not then get a chance to
    heal. Exactly one uniform draw is consumed per individual per year.

    See e.g. "The Association Between Parent and Child ACEs is Buffered by
    Forgiveness of Others and Self-Forgiveness", J Child Adolesc Trauma (2023).
    """
    if individual.age < 18:
        if np.random.random() < probability_of_trauma:
            return min(8, individual.number_of_aces + 1) - individual.number_of_aces
        return 0
    if np.random.random() < probability_of_heal:
        return max(0, individual.number_of_aces - 1) - individual.number_of_aces
    return 0
