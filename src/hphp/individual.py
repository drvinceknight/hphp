from typing import NamedTuple


class Individual(NamedTuple):
    """
    A member of the simulated population.

    Attributes
    ----------
    sex : str
        "Male" or "Female".
    age : int
        Age in whole years.
    number_of_aces : int
        Current adverse childhood experience count, which healing reduces.
    completed_number_of_aces : int
        The number of adverse childhood experiences accumulated by the end of
        childhood, frozen once the individual reaches 18. Since healing acts
        only on adults, this is the total an individual was ever exposed to,
        whereas `number_of_aces` is what remains after any recovery.

        The tempo effect reads this rather than the current count, because the
        life-history pace is calibrated during development and is not reset by
        later recovery. Using the current count would let an individual who
        had aged out of a compressed fertility window re-enter it by healing,
        which manufactures fertility rather than retiming it.
    """

    sex: str
    age: int
    number_of_aces: int
    completed_number_of_aces: int
