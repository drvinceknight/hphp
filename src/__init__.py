import random


def birth(age, number_of_aces=0, alpha=3):
    """
    This uses data from the UN's department of economic and social affairs
    - <https://population.un.org/wpp/Download/Standard/Fertility/>
    to return a probability of having a birth at a given age.

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
