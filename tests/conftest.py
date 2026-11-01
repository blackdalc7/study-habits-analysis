import pytest

from study_habits.clean import clean
from study_habits.simulate import make_messy, simulate


@pytest.fixture(scope="session")
def tidy():
    return simulate(400, seed=2026)


@pytest.fixture(scope="session")
def messy(tidy):
    return make_messy(tidy)


@pytest.fixture(scope="session")
def cleaned(messy):
    df, _ = clean(messy)
    return df
