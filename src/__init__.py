import random

def death(age):
    """
    This uses data from the UN's department of economic and social affairs
    https://population.un.org/wpp/Download/Standard/Mortality/
    to return a boolean representing whether or not someone has a death at a
    given age.

    Life Tables - Abridged

    """
    probability_of_death_at_give_age = [
        0.02793,
        0.00351,
        0.00244,
        0.00193,
        0.00160,
        0.00137,
        0.00118,
        0.00101,
        0.00087,
        0.00076,
        0.00069,
        0.00066,
        0.00065,
        0.00069,
        0.00077,
        0.00088,
        0.00100,
        0.00113,
        0.00125,
        0.00135,
        0.00142,
        0.00148,
        0.00153,
        0.00156,
        0.00159,
        0.00162,
        0.00165,
        0.00168,
        0.00172,
        0.00177,
        0.00180,
        0.00185,
        0.00194,
        0.00202,
        0.00212,
        0.00225,
        0.00239,
        0.00254,
        0.00268,
        0.00283,
        0.00303,
        0.00323,
        0.00345,
        0.00368,
        0.00391,
        0.00415,
        0.00441,
        0.00470,
        0.00506,
        0.00548,
        0.00596,
        0.00651,
        0.00711,
        0.00777,
        0.00847,
        0.00907,
        0.00975,
        0.01046,
        0.01141,
        0.01291,
        0.01438,
        0.01559,
        0.01678,
        0.01794,
        0.01937,
        0.02093,
        0.02256,
        0.02439,
        0.02629,
        0.02849,
        0.03112,
        0.03383,
        0.03705,
        0.03962,
        0.04307,
        0.04713,
        0.05090,
        0.05468,
        0.05940,
        0.06524,
        0.07183,
        0.07917,
        0.08707,
        0.09573,
        0.10509,
        0.11405,
        0.12365,
        0.13305,
        0.14272,
        0.15315,
        0.16429,
        0.17683,
        0.19091,
        0.20629,
        0.22148,
        0.23683,
        0.25414,
        0.27072,
        0.28827,
        0.30734,
        1.0000,
    ]
    try:
        probability_of_death = probability_of_death_at_give_age[age]
    except IndexError:
        probability_of_death = 1
    return random.random() < probability_of_death


def birth(age, number_of_aces=0, alpha=3):
    """
    This uses data from the UN's department of economic and social affairs
    - <https://population.un.org/wpp/Download/Standard/Fertility/>
    to return a boolean representing whether or not someone (able to have a
    birth) has a birth at a given age.

    - age: the age
    - number_of_aces: the number of adverse childhood events
    - alpha: an adjustment for the number of adverse childhood events (this is
      described in more detail in the adjust_age_for_aces docstring)
    """
    age = adjust_age_for_aces(age=age, number_of_aces=number_of_aces, alpha=alpha)
    overall_probability_of_a_birth_at_given_age = {
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
    probability = overall_probability_of_a_birth_at_given_age.get(age, 0)

    return random.random() < probability


def adjust_age_for_aces(age, number_of_aces, alpha=3):
    """
    This adjusts the probability based on the number of aces.

    This is an approximation but is based on the first table from
    'Adverse Childhood Experiences, Early and Nonmarital Fertility, and Women’s Health at Midlife'

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

    Whilst not directly and without further specific study this suggests the
    following adjustment:

    p(age) = p(age + alpha * number_of_aces)

    Where alpha is an adjustment that is set to 3.

    This is very approximate but suggests modifying the probability in that way.
    """
    return age + alpha * number_of_aces
