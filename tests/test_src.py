import random
import numpy as np
import src


def test_number_of_births_in_given_year_for_ages_out_of_data_set():
    number_of_repetitions = 1_000
    for age in range(15):
        assert  all([src.birth(age=age) is False for _ in
            range(number_of_repetitions)])
    for age in range(50, 100):
        assert  all([src.birth(age=age) is False for _ in
            range(number_of_repetitions)])


def test_number_of_births_in_given_year_for_ages_inside_range():
    number_of_repetitions = 1_000
    for age in range(15, 50):
        assert  any([src.birth(age=age) is True for _ in
            range(number_of_repetitions)])


def test_get_expected_number_of_births_for_specific_25_year_olds():
    """
    From the data we have that 1000 25 year olds give birth to approximately
    134.492 people.

    This is a wide ranging test that repeats the exercise 500 times and confirms an
    expectation.
    """
    expected_value = 134.492
    births = []
    for seed in range(500):
        random.seed(seed)
        number_of_repetitions = 1_000
        births.append(sum(src.birth(age=25) for _ in range(number_of_repetitions)))
    assert expected_value - 1 <= np.mean(births) <= expected_value + 1
