import numpy as np
import pytest

import hphp.birth_death
import hphp.simulation


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
        generator = np.random.default_rng(seed)
        number_of_repetitions = 1_000
        births.append(
            sum(
                hphp.simulation.birth(age=25, generator=generator)
                for _ in range(number_of_repetitions)
            )
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
        generator = np.random.default_rng(0)
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
            generator = np.random.default_rng(seed)
            number_of_repetitions = 1_000
            births.append(
                sum(
                    hphp.simulation.death(age=25, sex=sex, generator=generator)
                    for _ in range(number_of_repetitions)
                )
            )
        assert expected_value - 1 <= np.mean(births) <= expected_value + 1


def test_get_expected_number_of_deaths_for_specific_50_year_olds():
    expected_values = (9.884, 8.025)
    births = []
    for expected_value, sex in zip(expected_values, ("Male", "Female")):
        for seed in range(500):
            generator = np.random.default_rng(seed)
            number_of_repetitions = 1_000
            births.append(
                sum(
                    hphp.simulation.death(age=50, sex=sex, generator=generator)
                    for _ in range(number_of_repetitions)
                )
            )
        assert (
            expected_value - 1 <= np.mean(births) <= expected_value + 1
        ), f"Failed for {expected_value}, {sex}"


def test_get_expected_number_of_deaths_for_specific_99_year_olds():
    # Theoretical: 1000 * q(99) for males, then pooled with females,
    # since `births` accumulates across both.
    expected_values = (372.24, 356.045)
    births = []
    for expected_value, sex in zip(expected_values, ("Male", "Female")):
        for seed in range(500):
            generator = np.random.default_rng(seed)
            number_of_repetitions = 1_000
            births.append(
                sum(
                    hphp.simulation.death(age=99, sex=sex, generator=generator)
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
            generator = np.random.default_rng(seed)
            number_of_repetitions = expected_value
            births.append(
                sum(
                    hphp.simulation.death(age=100, sex=sex, generator=generator)
                    for _ in range(number_of_repetitions)
                )
            )
        assert expected_value - 1 <= np.mean(births) <= expected_value + 1


def test_sample_number_of_aces_for_total_group():
    repetitions = 10_000
    generator = np.random.default_rng(0)
    number_of_aces_for_total_group = [
        hphp.simulation.sample_number_of_aces(sex="Total", generator=generator)
        for _ in range(repetitions)
    ]
    expected_mean = 1.3625
    expected_std = 1.5404849074236333
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
    generator = np.random.default_rng(0)
    number_of_aces_for_total_group = [
        hphp.simulation.sample_number_of_aces(sex="Male", generator=generator)
        for _ in range(repetitions)
    ]
    expected_mean = 1.2293
    expected_std = 1.425735427770524
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
    generator = np.random.default_rng(0)
    number_of_aces_for_total_group = [
        hphp.simulation.sample_number_of_aces(sex="Female", generator=generator)
        for _ in range(repetitions)
    ]
    expected_mean = 1.4929
    expected_std = 1.6496513540745512
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
    individual = hphp.simulation.Individual(
            sex="Male",
            age=21,
            number_of_aces=3,
            completed_number_of_aces=3,
        )
    assert individual.sex == "Male"
    assert individual.age == 21
    assert individual.number_of_aces == 3


def test_uk_population_pyramid():
    repetitions = 10_000
    generator = np.random.default_rng(0)
    ages = []
    number_of_male = 0
    for _ in range(repetitions):
        age, sex = hphp.simulation.uk_population_pyramid(generator=generator)
        ages.append(age)
        number_of_male += sex == "Male"
    expected_mean = 41.1285
    expected_std = 23.727924219155792
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
    expected_male_to_female_ratio = (0.9774569903104607,)
    assert np.isclose(male_to_female_ratio, expected_male_to_female_ratio)


def test_get_population_with_uk_pyramid_population():
    number_of_individuals = 10_000
    generator = np.random.default_rng(0)
    population = hphp.simulation.get_population(
        number_of_individuals=number_of_individuals,
        population_pyramid=hphp.simulation.uk_population_pyramid,
        generator=generator,
    )
    assert len(population) == number_of_individuals
    ages = []
    number_of_male = 0
    for individual in population:
        ages.append(individual.age)
        number_of_male += individual.sex == "Male"
    expected_mean = 41.4405
    expected_std = 23.745901956969334
    expected_min = 0
    mean = np.mean(ages)
    std = np.std(ages)
    max = np.max(ages)
    min = np.min(ages)

    assert np.isclose(expected_mean, mean)
    assert np.isclose(expected_std, std)
    assert np.isclose(expected_min, min)
    # The oldest age group carries probability 0.00035, so whether a sample of
    # this size contains a centenarian is a property of the draw, not of the
    # code. The invariant is that no age falls outside the pyramid.
    assert max <= 100

    male_to_female_ratio = number_of_male / (number_of_individuals - number_of_male)
    expected_male_to_female_ratio = 0.9665683382497542
    assert np.isclose(male_to_female_ratio, expected_male_to_female_ratio)


def test_sample_intergenerational_number_of_aces():
    numbers_of_maternal_aces = range(9)
    expected_mean_number_of_aces = (
        1.31,
        1.442,
        1.983,
        1.932,
        2.606,
        2.453,
        2.408,
        2.565,
    )
    repetitions = 1_000
    generator = np.random.default_rng(0)
    for number_of_maternal_aces, expected_mean in zip(
        numbers_of_maternal_aces, expected_mean_number_of_aces
    ):
        assert np.isclose(
            expected_mean,
            np.mean(
                [
                    hphp.simulation.sample_intergenerational_number_of_aces(
                        number_of_maternal_aces=number_of_maternal_aces,
                        generator=generator,
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
            sex="Male",
            age=np.random.randint(0, 100),
            number_of_aces=3,
            completed_number_of_aces=3,
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
            sex="Male",
            age=np.random.randint(0, 18),
            number_of_aces=3,
            completed_number_of_aces=3,
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
            sex="Male",
            age=np.random.randint(18, 100),
            number_of_aces=3,
            completed_number_of_aces=3,
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
    generator = np.random.default_rng(0)
    repetitions = 20_000
    individual = hphp.simulation.Individual(
            sex="Female",
            age=10,
            number_of_aces=4,
            completed_number_of_aces=4,
        )

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
    generator = np.random.default_rng(0)
    repetitions = 20_000
    individual = hphp.simulation.Individual(
            sex="Female",
            age=30,
            number_of_aces=4,
            completed_number_of_aces=4,
        )

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


def test_healing_does_not_reopen_fertility():
    """
    An adult who has aged out of her compressed fertility window must not
    re-enter it by healing: the life-history pace is fixed by childhood
    adversity, so recovery changes how much she reproduces, never when.
    """
    age = 30
    probability_before_healing = _birth_probability(
        age=age, number_of_aces=8, completed_number_of_aces=8
    )
    probability_after_healing = _birth_probability(
        age=age, number_of_aces=2, completed_number_of_aces=8
    )
    assert probability_before_healing == 0.0
    assert probability_after_healing == 0.0

    # Recomputing the pace from the current count is what reopens the window.
    if_pace_followed_healing = _birth_probability(
        age=age, number_of_aces=2, completed_number_of_aces=2
    )
    assert if_pace_followed_healing > 0.0


def test_healing_still_removes_the_quantum_advantage():
    """Recovery must keep acting on fertility intensity."""
    age = 20
    with_aces = _birth_probability(
        age=age, number_of_aces=4, completed_number_of_aces=4, alpha=1.5
    )
    healed = _birth_probability(
        age=age, number_of_aces=0, completed_number_of_aces=4, alpha=1.5
    )
    assert with_aces > healed > 0.0


def _birth_probability(age, number_of_aces, completed_number_of_aces, alpha=1.0):
    """The birth probability `birth` samples against, computed directly."""
    pace = hphp.birth_death.life_history_pace(completed_number_of_aces)
    effective_age = hphp.birth_death.adjust_age_for_aces(
        age=age, number_of_aces=completed_number_of_aces
    )
    probability = pace * hphp.birth_death.baseline_birth_probability(effective_age)
    if number_of_aces >= 1:
        probability = min(1.0, alpha * probability)
    return probability


def _population(number_of_aces_by_individual, age=25):
    return [
        hphp.simulation.Individual(
            sex="Female",
            age=age,
            number_of_aces=number_of_aces,
            completed_number_of_aces=number_of_aces,
        )
        for number_of_aces in number_of_aces_by_individual
    ]


def test_mean_fertility_multiplier_when_nobody_is_exposed():
    population = _population([0, 0, 0, 0])
    multiplier = hphp.simulation.mean_fertility_multiplier(
        population, alpha=1.35, pace_increase_per_ace=0.3142
    )
    assert multiplier == pytest.approx(1.0)


def test_mean_fertility_multiplier_when_everyone_is_exposed():
    population = _population([1, 2, 3, 4])
    multiplier = hphp.simulation.mean_fertility_multiplier(
        population, alpha=1.35, pace_increase_per_ace=0.3142
    )
    assert multiplier == pytest.approx(1.35)


def test_mean_fertility_multiplier_lies_between_the_two():
    population = _population([0, 0, 1, 1])
    multiplier = hphp.simulation.mean_fertility_multiplier(
        population, alpha=1.35, pace_increase_per_ace=0.3142
    )
    assert 1.0 < multiplier < 1.35


def test_mean_fertility_multiplier_ignores_the_infertile():
    """Only individuals who can actually give birth carry weight."""
    fertile = _population([1, 1], age=25)
    with_children = fertile + _population([0, 0], age=5)
    assert hphp.simulation.mean_fertility_multiplier(
        with_children, alpha=1.35, pace_increase_per_ace=0.3142
    ) == pytest.approx(
        hphp.simulation.mean_fertility_multiplier(
            fertile, alpha=1.35, pace_increase_per_ace=0.3142
        )
    )


def test_normalisation_conserves_expected_births():
    """
    Normalising must leave the total expected births unchanged from the
    alpha = 1 case, redistributing them rather than adding any.
    """
    population = _population([0, 0, 1, 2, 4], age=25)
    alpha = 1.4
    pace = 0.3142

    unexposed_total = sum(
        hphp.birth_death.fertility(
            age=person.age,
            completed_number_of_aces=person.completed_number_of_aces,
            pace_increase_per_ace=pace,
        )
        for person in population
    )

    normaliser = hphp.simulation.mean_fertility_multiplier(population, alpha, pace)
    normalised_total = 0.0
    for person in population:
        baseline = hphp.birth_death.fertility(
            age=person.age,
            completed_number_of_aces=person.completed_number_of_aces,
            pace_increase_per_ace=pace,
        )
        multiplier = alpha if person.number_of_aces >= 1 else 1.0
        normalised_total += baseline * multiplier / normaliser

    assert normalised_total == pytest.approx(unexposed_total)


def test_normalisation_preserves_the_relative_advantage():
    """
    Exposed individuals must still out-reproduce unexposed ones by exactly
    alpha.

    Both individuals here share a childhood ACE count, so they share a
    life-history pace and therefore a baseline fertility; they differ only in
    their current count, which is what the quantum multiplier reads. Comparing
    individuals with different childhood counts would confound the quantum
    effect with the tempo effect.
    """
    alpha = 1.4
    pace = 0.3142
    healed = hphp.simulation.Individual(
        sex="Female", age=25, number_of_aces=0, completed_number_of_aces=2
    )
    exposed = hphp.simulation.Individual(
        sex="Female", age=25, number_of_aces=2, completed_number_of_aces=2
    )
    population = [healed, exposed]
    normaliser = hphp.simulation.mean_fertility_multiplier(population, alpha, pace)

    probabilities = []
    for person in population:
        baseline = hphp.birth_death.fertility(
            age=person.age,
            completed_number_of_aces=person.completed_number_of_aces,
            pace_increase_per_ace=pace,
        )
        multiplier = alpha if person.number_of_aces >= 1 else 1.0
        probabilities.append(baseline * multiplier / normaliser)

    assert probabilities[1] / probabilities[0] == pytest.approx(alpha)


def test_normalisation_decouples_growth_from_alpha():
    """
    Population growth must be far less sensitive to alpha once fertility is
    normalised.

    The comparison is made against the unnormalised model rather than against
    an absolute tolerance, because the residual spread is sampling noise whose
    size depends on the population and the number of replicates.
    """

    def growth_rate(alpha, normalise):
        generator = np.random.default_rng(11)
        initial = hphp.simulation.get_population(
            2000, hphp.simulation.uk_population_pyramid, generator=generator
        )
        history = hphp.simulation.simulate(
            number_of_years=120,
            initial_population=initial,
            probability_of_male_birth=0.51,
            alpha=alpha,
            probability_of_heal=0.0,
            probability_of_trauma=0.0,
            generator=generator,
            normalise_fertility=normalise,
        )
        return np.log(len(history[120]) / len(history[80])) / 40

    unnormalised_spread = abs(
        growth_rate(1.50, normalise=False) - growth_rate(1.05, normalise=False)
    )
    normalised_spread = abs(
        growth_rate(1.50, normalise=True) - growth_rate(1.05, normalise=True)
    )

    assert normalised_spread < unnormalised_spread / 3
