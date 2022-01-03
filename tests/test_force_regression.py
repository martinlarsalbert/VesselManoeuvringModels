import pytest
from src.models.regression import ForceRegression
import src.models.vmm_linear as vmm
import numpy as np
import pandas as pd
from src.prime_system import PrimeSystem


@pytest.fixture
def data():

    N = 30
    columns = ["fx", "fy", "mz", "u", "v", "r", "delta"]
    d = np.random.normal(size=(N, len(columns)))  # random data
    df = pd.DataFrame(data=d, columns=columns)
    yield df


@pytest.fixture
def added_masses():
    added_masses_ = {
        "Xudot": 1.0,
        "Yvdot": 1.0,
        "Nrdot": 1.0,
        "Yrdot": 1.0,
        "Nvdot": 1.0,
    }
    df = pd.DataFrame(added_masses_, index=["prime"]).transpose()
    yield df


@pytest.fixture
def ship_parameters():
    s = {
        "L": 1,
        "rho": 1,
    }
    yield s


@pytest.fixture
def prime_system(ship_parameters):
    ps = PrimeSystem(**ship_parameters)
    yield ps


def test_force_regression(data):

    regression = ForceRegression(vmm=vmm, data=data)


def test_force_regression_create_model(
    data, added_masses, prime_system, ship_parameters
):

    regression = ForceRegression(vmm=vmm, data=data)

    model = regression.create_model(
        added_masses=added_masses,
        ship_parameters=ship_parameters,
        ps=prime_system,
        control_keys=["delta"],
    )
