import random
import numpy as np
import src


def test_number_of_births_in_given_year_for_ages_out_of_data_set():
    number_of_repetitions = 5_000
    for age in range(15):
        assert all([src.birth(age=age) is False for _ in range(number_of_repetitions)])
    for age in range(50, 100):
        assert all([src.birth(age=age) is False for _ in range(number_of_repetitions)])


def test_number_of_births_in_given_year_for_ages_inside_range_gives_a_birth():
    number_of_repetitions = 5_000
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


def test_number_of_deaths_in_given_year_for_ages_outside_range_gives_a_death():
    number_of_repetitions = 100_000
    for age in range(101, 105):
        assert all([src.death(age=age) is True for _ in range(number_of_repetitions)])


def test_number_of_deaths_in_given_year_for_ages_inside_range_gives_a_death():
    number_of_repetitions = 100_000
    for age in range(101):
        assert any([src.death(age=age) is True for _ in range(number_of_repetitions)])


def test_get_expected_number_of_deaths_for_specific_25_year_olds():
    """
    From the data we have that 94_205 individuals are expected to survive to 25.

    Of those, 152.4 are expected to die.

    This is a wide ranging test that repeats the exercise 500 times and confirms an
    expectation.
    """
    expected_value = 152.4
    births = []
    for seed in range(500):
        random.seed(seed)
        number_of_repetitions = 94_205
        births.append(sum(src.death(age=25) for _ in range(number_of_repetitions)))
    assert expected_value - 1 <= np.mean(births) <= expected_value + 1


def test_get_expected_number_of_deaths_for_specific_50_year_olds():
    """
    From the data we have that 87_655 individuals are expected to survive to 50.

    Of those, 522.5 are expected to die.

    This is a wide ranging test that repeats the exercise 500 times and confirms an
    expectation.
    """
    expected_value = 522.5
    births = []
    for seed in range(500):
        random.seed(seed)
        number_of_repetitions = 87_655
        births.append(sum(src.death(age=50) for _ in range(number_of_repetitions)))
    assert expected_value - 1 <= np.mean(births) <= expected_value + 1


def test_get_expected_number_of_deaths_for_specific_99_year_olds():
    """
    From the data we have that 1_269 individuals are expected to survive to 99.

    Of those, 390.1 are expected to die.

    This is a wide ranging test that repeats the exercise 500 times and confirms an
    expectation.
    """
    expected_value = 390.1
    births = []
    for seed in range(500):
        random.seed(seed)
        number_of_repetitions = 1_269
        births.append(sum(src.death(age=99) for _ in range(number_of_repetitions)))
    assert expected_value - 1 <= np.mean(births) <= expected_value + 1


def test_get_expected_number_of_deaths_for_specific_100_year_olds():
    """
    All 100 year olds die (although the data set isn't completely clear here
    except for the probability of death being 1)
    """
    expected_value = 879
    births = []
    for seed in range(500):
        random.seed(seed)
        number_of_repetitions = expected_value
        births.append(sum(src.death(age=100) for _ in range(number_of_repetitions)))
    assert expected_value - 1 <= np.mean(births) <= expected_value + 1
