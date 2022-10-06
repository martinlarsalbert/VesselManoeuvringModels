import vessel_manoeuvring_models.models.linear_vmm as model
import pytest
import pandas as pd
import numpy as np
import vessel_manoeuvring_models.prime_system
from vessel_manoeuvring_models.models import brix_coefficients
from vessel_manoeuvring_models.prime_system import PrimeSystem

from pandas.testing import assert_frame_equal
import matplotlib.pyplot as plt


@pytest.fixture
def ship_parameters():
    T_ = 10
    L_ = 200
    CB_ = 0.7
    B_ = 30
    rho_ = 1025
    volume_ = T_ * B_ * L_ * CB_
    m_ = volume_ * rho_

    yield {
        "T": T_,
        "L": L_,
        "CB": CB_,
        "B": B_,
        "rho": rho_,
        "x_G": 0,
        "m": m_,
        "I_z": 0.2 * m_ * L_ ** 2,
        "volume": volume_,
    }


@pytest.fixture
def df_ship_parameters(ship_parameters):

    df_ship_parameters = pd.DataFrame(data=ship_parameters, index=["value"])
    ps = PrimeSystem(**ship_parameters)
    df_ship_parameters.loc["prime"] = ps.prime(ship_parameters)

    yield df_ship_parameters


@pytest.fixture
def df_parameters(df_ship_parameters):

    df_parameters = brix_coefficients.calculate(df_ship_parameters=df_ship_parameters)
    df_parameters["prime"].fillna(0, inplace=True)
    df_parameters.loc["Ydelta", "prime"] = 0.1  # Just guessing
    df_parameters.loc["Ndelta", "prime"] = 0.1  # Just guessing

    yield df_parameters


@pytest.fixture
def prime_system(ship_parameters):
    yield vessel_manoeuvring_models.prime_system.PrimeSystem(**ship_parameters)


def test_sim1(ship_parameters, df_parameters):

    t = np.linspace(0, 10, 100)
    df_ = pd.DataFrame(index=t)
    df_["u"] = 10
    df_["v"] = 0
    df_["r"] = 0
    df_["x0"] = 0
    df_["y0"] = 0
    df_["psi"] = 0
    df_["delta"] = 0

    parameters = df_parameters["prime"]
    # (This works but is acutally wrong since prime parameters and ship_parameters in SI are mixed)
    result = model.simulator.simulate(
        df_=df_,
        parameters=parameters,
        ship_parameters=ship_parameters,
        control_keys=["delta"],
    )


def test_primed_parameters(ship_parameters, df_parameters, prime_system):

    t = np.linspace(0, 10, 100)
    df_ = pd.DataFrame(index=t)
    df_["u"] = 10
    df_["v"] = 0
    df_["r"] = 0
    df_["x0"] = 0
    df_["y0"] = 0
    df_["psi"] = 0
    df_["delta"] = 0
    df_["U"] = np.sqrt(df_["u"] ** 2 + df_["v"] ** 2)

    parameters = df_parameters["prime"]
    # primed_parameters=True will get primed parameters to work in an SI unit world.
    result = model.simulator.simulate(
        df_=df_,
        parameters=parameters,
        ship_parameters=ship_parameters,
        control_keys=["delta"],
        primed_parameters=True,
        prime_system=prime_system,
    )


def test_resimulated(ship_parameters, df_parameters, prime_system):
    N = 100
    t = np.linspace(0, 10, N)
    df_ = pd.DataFrame(index=t)
    df_["u"] = 10
    df_["v"] = 0
    df_["r"] = 0
    df_["x0"] = 0
    df_["y0"] = 0
    df_["psi"] = 0
    df_["delta"] = np.deg2rad(
        np.concatenate(
            (
                np.zeros(int(N / 4)),
                np.ones(int(N / 4)) * 40,
                np.ones(int(N / 4)) * -40,
                np.zeros(int(N / 4)),
            )
        )
    )
    df_["U"] = np.sqrt(df_["u"] ** 2 + df_["v"] ** 2)

    parameters = df_parameters["prime"]
    # primed_parameters=True will get primed parameters to work in an SI unit world.
    result = model.simulator.simulate(
        df_=df_,
        parameters=parameters,
        ship_parameters=ship_parameters,
        control_keys=["delta"],
        primed_parameters=True,
        prime_system=prime_system,
    )

    result2 = model.simulator.simulate(
        df_=result.result,
        parameters=parameters,
        ship_parameters=ship_parameters,
        control_keys=["delta"],
        primed_parameters=True,
        prime_system=prime_system,
    )

    # result2.plot()
    # plt.show()

    assert_frame_equal(
        result.result, result2.result, check_exact=False, check_less_precise=5
    )
