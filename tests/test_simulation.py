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


def test_life_history_pace():
    assert hphp.birth_death.life_history_pace(number_of_aces=0) == 1.0
    assert hphp.birth_death.life_history_pace(
        number_of_aces=2, pace_increase_per_ace=0.25
    ) == 1.5


def test_adjustment_for_age():
    """
    The tempo effect accelerates the life-history schedule, so an exposed
    individual is evaluated at an older effective age and reaches any given
    point of the schedule at a younger real age.
    """
    age = 15
    assert hphp.birth_death.adjust_age_for_aces(age=age, number_of_aces=0) == age
    assert hphp.birth_death.adjust_age_for_aces(age=age, number_of_aces=4) == age

    age = 35
    unexposed = hphp.birth_death.adjust_age_for_aces(age=age, number_of_aces=0)
    exposed = hphp.birth_death.adjust_age_for_aces(age=age, number_of_aces=1)
    assert unexposed == age
    assert exposed > age

    assert hphp.birth_death.adjust_age_for_aces(
        age=25, number_of_aces=2, pace_increase_per_ace=0.5
    ) == 15 + (25 - 15) * 2.0


def test_fertile_window_never_starts_below_youngest_fertile_age():
    """
    A rigid translation of the schedule slides the fertile window below the
    youngest age in the schedule; rescaling about that age cannot.
    """
    for number_of_aces in range(9):
        youngest = next(
            age
            for age in range(101)
            if hphp.birth_death.baseline_birth_probability(
                hphp.birth_death.adjust_age_for_aces(
                    age=age, number_of_aces=number_of_aces
                )
            )
            > 0
        )
        assert youngest >= hphp.birth_death.YOUNGEST_FERTILE_AGE


def test_completed_fertility_is_preserved():
    """
    The manuscript claims the tempo effect is a purely timing adjustment, so
    the expected number of births must not change with ACE count at alpha = 1.
    """
    baseline = _expected_births(number_of_aces=0)
    for number_of_aces in range(1, 9):
        assert abs(_expected_births(number_of_aces) / baseline - 1) < 0.05


def test_mean_age_at_birth_falls_with_aces():
    """
    The model claim under test: ACE exposure shifts childbearing earlier, by
    about three years for the first ACE.
    """
    mean_ages = [_mean_age_at_birth(number_of_aces) for number_of_aces in range(9)]

    assert mean_ages == sorted(mean_ages, reverse=True)
    assert abs((mean_ages[0] - mean_ages[1]) - 3.0) < 0.15


def _birth_probabilities(number_of_aces):
    pace = hphp.birth_death.life_history_pace(number_of_aces)
    return [
        pace
        * hphp.birth_death.baseline_birth_probability(
            hphp.birth_death.adjust_age_for_aces(
                age=age, number_of_aces=number_of_aces
            )
        )
        for age in range(101)
    ]


def _expected_births(number_of_aces):
    return sum(_birth_probabilities(number_of_aces))


def _mean_age_at_birth(number_of_aces):
    probabilities = _birth_probabilities(number_of_aces)
    total = sum(probabilities)
    return sum(age * p for age, p in enumerate(probabilities)) / total


def test_get_expected_difference_for_20_year_olds():
    """
    This tests that aces impact on births in the expected way.

    For 20 year olds (below the peak of the schedule at 25): the number of
    births increases with the number of aces, because the accelerated schedule
    brings the peak forward.
    """
    age = 20
    number_of_repetitions = 1_000

    counts = []
    for number_of_aces in range(4):
        np.random.seed(0)
        counts.append(
            sum(
                hphp.simulation.birth(age=age, number_of_aces=number_of_aces)
                for _ in range(number_of_repetitions)
            )
        )

    assert counts == sorted(counts)
    assert counts[3] > counts[0]


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


def test_children_never_heal():
    """
    Healing is an adult process, so a child must never lose an ACE, whatever
    the healing probability.

    Both probabilities are positive here: with either one switched off the
    child and adult branches cannot interact, which is what allowed a
    fall-through from the trauma branch into the healing branch to go
    unnoticed.
    """
    np.random.seed(0)
    repetitions = 20_000
    individual = hphp.simulation.Individual(sex="Female", age=10, number_of_aces=4)

    deltas = [
        hphp.simulation.adjust_aces(
            individual, probability_of_heal=0.5, probability_of_trauma=0.2
        )
        for _ in range(repetitions)
    ]

    assert min(deltas) == 0
    assert max(deltas) == 1

    proportion_gaining = sum(delta == 1 for delta in deltas) / repetitions
    assert 0.19 < proportion_gaining < 0.21


def test_adults_heal_at_the_healing_probability():
    """An adult is unaffected by the trauma probability."""
    np.random.seed(0)
    repetitions = 20_000
    individual = hphp.simulation.Individual(sex="Female", age=30, number_of_aces=4)

    deltas = [
        hphp.simulation.adjust_aces(
            individual, probability_of_heal=0.5, probability_of_trauma=0.2
        )
        for _ in range(repetitions)
    ]

    assert min(deltas) == -1
    assert max(deltas) == 0

    proportion_healing = sum(delta == -1 for delta in deltas) / repetitions
    assert 0.49 < proportion_healing < 0.51


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
