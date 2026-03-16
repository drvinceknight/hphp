import numpy as np
import hphp.simulation
import hphp.birth_death


def test_number_of_births_in_given_year_for_ages_out_of_data_set():
    number_of_repetitions = 5_000
    for age in range(15):
        assert all(
            [
                hphp.simulation.birth(age=age) is False
                for _ in range(number_of_repetitions)
            ]
        )
    for age in range(50, 100):
        assert all(
            [
                hphp.simulation.birth(age=age) is False
                for _ in range(number_of_repetitions)
            ]
        )


def test_number_of_births_in_given_year_for_ages_inside_range_gives_a_birth():
    number_of_repetitions = 5_000
    for age in range(15, 50):
        assert any(
            [
                hphp.simulation.birth(age=age) is True
                for _ in range(number_of_repetitions)
            ]
        )


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
        np.random.seed(seed)
        number_of_repetitions = 1_000
        births.append(
            sum(hphp.simulation.birth(age=25) for _ in range(number_of_repetitions))
        )
    assert expected_value - 1 <= np.mean(births) <= expected_value + 1


def test_adjustment_for_age():
    age = 15
    adjusted_age = hphp.birth_death.adjust_age_for_aces(age=age, number_of_aces=0)
    assert age == adjusted_age

    age = 15
    adjusted_age = hphp.birth_death.adjust_age_for_aces(age=age, number_of_aces=1)
    assert age - 3 == adjusted_age

    age = 15
    adjusted_age = hphp.birth_death.adjust_age_for_aces(
        age=age, number_of_aces=1, tempo_years_per_ace=2
    )
    assert age - 2 == adjusted_age

    age = 25
    adjusted_age = hphp.birth_death.adjust_age_for_aces(
        age=age, number_of_aces=5, tempo_years_per_ace=2
    )
    assert age - 10 == adjusted_age


def test_get_expected_difference_for_25_year_olds():
    """
    This tests that aces impact on births in the expected way.

    For 25 year olds: the number of births decreases with the number of aces.
    """
    age = 25
    np.random.seed(0)
    number_of_repetitions = 1_000
    number_of_aces = 0
    assert (
        sum(
            hphp.simulation.birth(age=age, number_of_aces=number_of_aces)
            for _ in range(number_of_repetitions)
        )
        == 139
    )

    number_of_aces = 1
    assert (
        sum(
            hphp.simulation.birth(age=age, number_of_aces=number_of_aces)
            for _ in range(number_of_repetitions)
        )
        == 132
    )

    number_of_aces = 2
    assert (
        sum(
            hphp.simulation.birth(age=age, number_of_aces=number_of_aces)
            for _ in range(number_of_repetitions)
        )
        == 69
    )

    number_of_aces = 3
    assert (
        sum(
            hphp.simulation.birth(age=age, number_of_aces=number_of_aces)
            for _ in range(number_of_repetitions)
        )
        == 26
    )

    number_of_aces = 4
    assert (
        sum(
            hphp.simulation.birth(age=age, number_of_aces=number_of_aces)
            for _ in range(number_of_repetitions)
        )
        == 0
    )

    number_of_aces = 5
    assert (
        sum(
            hphp.simulation.birth(age=age, number_of_aces=number_of_aces)
            for _ in range(number_of_repetitions)
        )
        == 0
    )

    number_of_aces = 6
    assert (
        sum(
            hphp.simulation.birth(age=age, number_of_aces=number_of_aces)
            for _ in range(number_of_repetitions)
        )
        == 0
    )


def test_number_of_deaths_in_given_year_for_ages_outside_range_gives_a_death_male():
    number_of_repetitions = 100_000
    for age in range(101, 105):
        assert all(
            [
                hphp.simulation.death(age=age, sex="Male") is True
                for _ in range(number_of_repetitions)
            ]
        )


def test_number_of_deaths_in_given_year_for_ages_outside_range_gives_a_death_female():
    number_of_repetitions = 100_000
    for age in range(101, 105):
        assert all(
            [
                hphp.simulation.death(age=age, sex="Female") is True
                for _ in range(number_of_repetitions)
            ]
        )


def test_number_of_deaths_in_given_year_for_ages_inside_range_gives_a_death_male():
    number_of_repetitions = 100_000
    for age in range(101):
        assert any(
            [
                hphp.simulation.death(age=age, sex="Male") is True
                for _ in range(number_of_repetitions)
            ]
        )


def test_number_of_deaths_in_given_year_for_ages_inside_range_gives_a_death_female():
    number_of_repetitions = 100_000
    for age in range(101):
        assert any(
            [
                hphp.simulation.death(age=age, sex="Female") is True
                for _ in range(number_of_repetitions)
            ]
        )


def test_get_expected_number_of_deaths_for_specific_25_year_olds():
    """
    From the data we have that 94_205 individuals are expected to survive to 25.

    Of those, 152.4 are expected to die.

    This is a wide ranging test that repeats the exercise 500 times and confirms an
    expectation.
    """
    expected_value = 3.068
    births = []
    for sex in ("Male", "Female"):
        for seed in range(500):
            np.random.seed(seed)
            number_of_repetitions = 1_000
            births.append(
                sum(
                    hphp.simulation.death(age=25, sex=sex)
                    for _ in range(number_of_repetitions)
                )
            )
        assert expected_value - 1 <= np.mean(births) <= expected_value + 1


def test_get_expected_number_of_deaths_for_specific_50_year_olds():
    expected_values = (9.884, 8.025)
    births = []
    for expected_value, sex in zip(expected_values, ("Male", "Female")):
        for seed in range(500):
            np.random.seed(seed)
            number_of_repetitions = 1_000
            births.append(
                sum(
                    hphp.simulation.death(age=50, sex=sex)
                    for _ in range(number_of_repetitions)
                )
            )
        assert (
            expected_value - 1 <= np.mean(births) <= expected_value + 1
        ), f"Failed for {expected_value}, {sex}"


def test_get_expected_number_of_deaths_for_specific_99_year_olds():
    expected_values = (372.892, 356.631)
    births = []
    for expected_value, sex in zip(expected_values, ("Male", "Female")):
        for seed in range(500):
            np.random.seed(seed)
            number_of_repetitions = 1_000
            births.append(
                sum(
                    hphp.simulation.death(age=99, sex=sex)
                    for _ in range(number_of_repetitions)
                )
            )
        assert (
            expected_value - 1 <= np.mean(births) <= expected_value + 1
        ), f"Failed for {expected_value}, {sex}"


def test_get_expected_number_of_deaths_for_specific_100_year_olds():
    expected_value = 879
    births = []
    for sex in ("Male", "Female"):
        for seed in range(500):
            np.random.seed(seed)
            number_of_repetitions = expected_value
            births.append(
                sum(
                    hphp.simulation.death(age=100, sex=sex)
                    for _ in range(number_of_repetitions)
                )
            )
        assert expected_value - 1 <= np.mean(births) <= expected_value + 1


def test_sample_number_of_aces_for_total_group():
    repetitions = 10_000
    np.random.seed(0)
    number_of_aces_for_total_group = [
        hphp.simulation.sample_number_of_aces(sex="Total") for _ in range(repetitions)
    ]
    expected_mean = 1.3451
    expected_std = 1.533820716381155
    expected_max = 8
    expected_min = 0
    mean = np.mean(number_of_aces_for_total_group)
    std = np.std(number_of_aces_for_total_group)
    max = np.max(number_of_aces_for_total_group)
    min = np.min(number_of_aces_for_total_group)
    assert np.isclose(expected_mean, mean)
    assert np.isclose(expected_std, std)
    assert np.isclose(expected_max, max)
    assert np.isclose(expected_min, min)


def test_number_of_aces_for_male_group():
    repetitions = 10_000
    np.random.seed(0)
    number_of_aces_for_total_group = [
        hphp.simulation.sample_number_of_aces(sex="Male") for _ in range(repetitions)
    ]
    expected_mean = 1.218
    expected_std = 1.4172071125985786
    expected_max = 8
    expected_min = 0
    mean = np.mean(number_of_aces_for_total_group)
    std = np.std(number_of_aces_for_total_group)
    max = np.max(number_of_aces_for_total_group)
    min = np.min(number_of_aces_for_total_group)
    assert np.isclose(expected_mean, mean)
    assert np.isclose(expected_std, std)
    assert np.isclose(expected_max, max)
    assert np.isclose(expected_min, min)


def test_number_of_aces_for_female_group():
    repetitions = 10_000
    np.random.seed(0)
    number_of_aces_for_total_group = [
        hphp.simulation.sample_number_of_aces(sex="Female") for _ in range(repetitions)
    ]
    expected_mean = 1.4829
    expected_std = 1.6409471624644105
    expected_max = 8
    expected_min = 0
    mean = np.mean(number_of_aces_for_total_group)
    std = np.std(number_of_aces_for_total_group)
    max = np.max(number_of_aces_for_total_group)
    min = np.min(number_of_aces_for_total_group)
    assert np.isclose(expected_mean, mean)
    assert np.isclose(expected_std, std)
    assert np.isclose(expected_max, max)
    assert np.isclose(expected_min, min)


def test_individual():
    individual = hphp.simulation.Individual(sex="Male", age=21, number_of_aces=3)
    assert individual.sex == "Male"
    assert individual.age == 21
    assert individual.number_of_aces == 3


def test_uk_population_pyramid():
    repetitions = 10_000
    np.random.seed(0)
    ages = []
    number_of_male = 0
    for _ in range(repetitions):
        age, sex = hphp.simulation.uk_population_pyramid()
        ages.append(age)
        number_of_male += sex == "Male"
    expected_mean = 41.0012
    expected_std = 23.66099741261978
    expected_max = 100
    expected_min = 0
    mean = np.mean(ages)
    std = np.std(ages)
    max = np.max(ages)
    min = np.min(ages)
    assert np.isclose(expected_mean, mean)
    assert np.isclose(expected_std, std)
    assert np.isclose(expected_max, max)
    assert np.isclose(expected_min, min)
    male_to_female_ratio = number_of_male / (repetitions - number_of_male)
    expected_male_to_female_ratio = (0.9657951641438962,)
    assert np.isclose(male_to_female_ratio, expected_male_to_female_ratio)


def test_get_population_with_uk_pyramid_population():
    number_of_individuals = 10_000
    population = hphp.simulation.get_population(
        number_of_individuals=number_of_individuals,
        population_pyramid=hphp.simulation.uk_population_pyramid,
        seed=0,
    )
    assert len(population) == number_of_individuals
    ages = []
    number_of_male = 0
    for individual in population:
        ages.append(individual.age)
        number_of_male += individual.sex == "Male"
    expected_mean = 41.1292
    expected_std = 23.736172972069443
    expected_max = 100
    expected_min = 0
    mean = np.mean(ages)
    std = np.std(ages)
    max = np.max(ages)
    min = np.min(ages)

    assert np.isclose(expected_mean, mean)
    assert np.isclose(expected_std, std)
    assert np.isclose(expected_max, max)
    assert np.isclose(expected_min, min)

    male_to_female_ratio = number_of_male / (number_of_individuals - number_of_male)
    expected_male_to_female_ratio = 0.9860973187686196
    assert np.isclose(male_to_female_ratio, expected_male_to_female_ratio)


def test_sample_intergenerational_number_of_aces():
    numbers_of_maternal_aces = range(9)
    expected_mean_number_of_aces = (
        1.248,
        1.64,
        2.053,
        1.893,
        2.383,
        2.456,
        2.39,
        2.482,
    )
    repetitions = 1_000
    np.random.seed(0)
    for number_of_maternal_aces, expected_mean in zip(
        numbers_of_maternal_aces, expected_mean_number_of_aces
    ):
        assert np.isclose(
            expected_mean,
            np.mean(
                [
                    hphp.simulation.sample_intergenerational_number_of_aces(
                        number_of_maternal_aces=number_of_maternal_aces
                    )
                    for _ in range(repetitions)
                ]
            ),
        )


def test_adjust_aces_with_no_probability():
    probability_of_heal = 0
    probability_of_trauma = 0
    repetitions = 100
    for _ in range(repetitions):
        individual = hphp.simulation.Individual(
            sex="Male", age=np.random.randint(0, 100), number_of_aces=3
        )
        delta = hphp.simulation.adjust_aces(
            individual,
            probability_of_heal=probability_of_heal,
            probability_of_trauma=probability_of_trauma,
        )
        assert delta == 0


def test_adjust_aces_for_children():
    probability_of_heal = 0
    probability_of_trauma = 0
    repetitions = 100
    for _ in range(repetitions):
        individual = hphp.simulation.Individual(
            sex="Male", age=np.random.randint(0, 18), number_of_aces=3
        )
        delta = hphp.simulation.adjust_aces(
            individual,
            probability_of_heal=probability_of_heal,
            probability_of_trauma=probability_of_trauma,
        )
        assert 1 >= delta >= 0


def test_adjust_aces_for_adults():
    probability_of_heal = 0
    probability_of_trauma = 0
    repetitions = 100
    for _ in range(repetitions):
        individual = hphp.simulation.Individual(
            sex="Male", age=np.random.randint(18, 100), number_of_aces=3
        )
        delta = hphp.simulation.adjust_aces(
            individual,
            probability_of_heal=probability_of_heal,
            probability_of_trauma=probability_of_trauma,
        )
        assert -1 <= delta <= 0


def test_get_initial_population():
    number_of_initial_individuals = 100
    seed = 0
    initial_population = hphp.simulation.get_population(
        number_of_individuals=number_of_initial_individuals,
        population_pyramid=hphp.simulation.uk_population_pyramid,
        seed=seed,
    )
    assert len(initial_population) == number_of_initial_individuals
    for individual in initial_population:
        assert individual.sex in ("Male", "Female")
        assert 0 <= individual.age <= 100
        assert 0 <= individual.number_of_aces <= 8
