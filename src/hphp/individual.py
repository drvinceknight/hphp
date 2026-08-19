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
    childhood_number_of_aces : int
        ACE count at the end of childhood, frozen once the individual reaches
        18. The tempo effect reads this rather than `number_of_aces`, because
        the life-history pace is calibrated during development and is not
        reset by later recovery. Using the current count instead would let an
        individual who has aged out of a compressed fertility window re-enter
        it by healing, which manufactures fertility rather than retiming it.
    """

    sex: str
    age: int
    number_of_aces: int
    childhood_number_of_aces: int
