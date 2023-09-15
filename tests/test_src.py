import random
import numpy as np
import src


def test_number_of_births_in_given_year_for_ages_out_of_data_set():
    number_of_repetitions = 1_000
    for age in range(15):
        assert all([src.birth(age=age) is False for _ in range(number_of_repetitions)])
    for age in range(50, 100):
        assert all([src.birth(age=age) is False for _ in range(number_of_repetitions)])


def test_number_of_births_in_given_year_for_ages_inside_range():
    number_of_repetitions = 1_000
    for age in range(15, 50):
        assert any([src.birth(age=age) is True for _ in range(number_of_repetitions)])


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


def test_adjustment_for_age():
    age = 15
    adjusted_age = src.adjust_age_for_aces(age=age, number_of_aces=0)
    assert age == adjusted_age

    age = 15
    adjusted_age = src.adjust_age_for_aces(age=age, number_of_aces=1)
    assert age + 3 == adjusted_age

    age = 15
    adjusted_age = src.adjust_age_for_aces(age=age, number_of_aces=1, alpha=2)
    assert age + 2 == adjusted_age

    age = 25
    adjusted_age = src.adjust_age_for_aces(age=age, number_of_aces=5, alpha=2)
    assert age + 10 == adjusted_age


def test_get_expected_difference_for_15_year_olds():
    """
    This tests that aces act in the expected way.

    For 15 year olds: the number of births should increase as the number of ACEs
    increases until we get to 5 aces.
    """
    age = 15
    random.seed(0)
    number_of_repetitions = 1_000
    number_of_aces = 0
    assert (
        sum(
            src.birth(age=age, number_of_aces=number_of_aces)
            for _ in range(number_of_repetitions)
        )
        == 18
    )

    number_of_aces = 1
    assert (
        sum(
            src.birth(age=age, number_of_aces=number_of_aces)
            for _ in range(number_of_repetitions)
        )
        == 64
    )

    number_of_aces = 2
    assert (
        sum(
            src.birth(age=age, number_of_aces=number_of_aces)
            for _ in range(number_of_repetitions)
        )
        == 117
    )

    number_of_aces = 3
    assert (
        sum(
            src.birth(age=age, number_of_aces=number_of_aces)
            for _ in range(number_of_repetitions)
        )
        == 122
    )

    number_of_aces = 4
    assert (
        sum(
            src.birth(age=age, number_of_aces=number_of_aces)
            for _ in range(number_of_repetitions)
        )
        == 130
    )

    number_of_aces = 5
    assert (
        sum(
            src.birth(age=age, number_of_aces=number_of_aces)
            for _ in range(number_of_repetitions)
        )
        == 116
    )

    number_of_aces = 6
    assert (
        sum(
            src.birth(age=age, number_of_aces=number_of_aces)
            for _ in range(number_of_repetitions)
        )
        == 84
    )


def test_get_expected_difference_for_25_year_olds():
    """
    This tests that aces act in the expected way.

    For 25 year olds: the number of decreases
    """
    age = 25
    random.seed(0)
    number_of_repetitions = 1_000
    number_of_aces = 0
    assert (
        sum(
            src.birth(age=age, number_of_aces=number_of_aces)
            for _ in range(number_of_repetitions)
        )
        == 148
    )

    number_of_aces = 1
    assert (
        sum(
            src.birth(age=age, number_of_aces=number_of_aces)
            for _ in range(number_of_repetitions)
        )
        == 124
    )

    number_of_aces = 2
    assert (
        sum(
            src.birth(age=age, number_of_aces=number_of_aces)
            for _ in range(number_of_repetitions)
        )
        == 112
    )

    number_of_aces = 3
    assert (
        sum(
            src.birth(age=age, number_of_aces=number_of_aces)
            for _ in range(number_of_repetitions)
        )
        == 69
    )

    number_of_aces = 4
    assert (
        sum(
            src.birth(age=age, number_of_aces=number_of_aces)
            for _ in range(number_of_repetitions)
        )
        == 52
    )

    number_of_aces = 5
    assert (
        sum(
            src.birth(age=age, number_of_aces=number_of_aces)
            for _ in range(number_of_repetitions)
        )
        == 30
    )

    number_of_aces = 6
    assert (
        sum(
            src.birth(age=age, number_of_aces=number_of_aces)
            for _ in range(number_of_repetitions)
        )
        == 16
    )
