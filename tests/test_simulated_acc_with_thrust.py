"""
The recalculated acceleration based on states from simulation does not seem to match.
This test reproduces the observed error, in the search for a bug fix...
This bug has been corrected to get this test to pass, the test will remain so that the bug does not 
reappear.
"""
import pytest
import src.models.vmm_simple_nonlinear as vmm

from src.parameters import df_parameters
from src.substitute_dynamic_symbols import run
from src import prime_system
import scipy.integrate
from numpy.testing import assert_almost_equal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ship_parameters = {
    "T": 0.2063106796116504,
    "L": 5.014563106796117,
    "CB": 0.45034232324249973,
    "B": 0.9466019417475728,
    "rho": 1000,
    "x_G": 0,
    "m": 441.0267843660858,
    "I_z": 693.124396594905,
    "volume": 0.4410267843660858,
}


def calculate_prime(row, ship_parameters):
    return run(function=row["brix_lambda"], **ship_parameters)


mask = df_parameters["brix_lambda"].notnull()
df_parameters.loc[mask, "brix_prime"] = df_parameters.loc[mask].apply(
    calculate_prime, ship_parameters=ship_parameters, axis=1
)

df_parameters["prime"] = df_parameters["brix_prime"]

df_parameters.loc["Ydelta", "prime"] = 0.001  # Just guessing
df_parameters.loc["Ndelta", "prime"] = (
    -df_parameters.loc["Ydelta", "prime"] / 2
)  # Just guessing

df_parameters.loc["Nur", "prime"] = 0.0001
df_parameters.loc["Nvrr", "prime"] = 0.0001
df_parameters.loc["Nvvr", "prime"] = 0.0001
df_parameters.loc["Xdeltadelta", "prime"] = -0.0001
df_parameters.loc["Xrr", "prime"] = 0.0025
df_parameters.loc["Xuu", "prime"] = -0.001
df_parameters.loc["Xvr", "prime"] = -0.001
df_parameters.loc["Xvv", "prime"] = -0.001
df_parameters.loc["Yur", "prime"] = 0.001
df_parameters.loc["Yvrr", "prime"] = 0.001
df_parameters.loc["Yvvr", "prime"] = 0
df_parameters.loc["Xthrust", "prime"] = 1

ps = prime_system.PrimeSystem(**ship_parameters)  # model
ship_parameters_prime = ps.prime(ship_parameters)


def simulate(thrust_):
    parameters = df_parameters["prime"].copy()
    t_ = np.linspace(0, 5, 100)
    df_ = pd.DataFrame(index=t_)

    df_["u"] = 2
    df_["v"] = 0
    df_["r"] = 0
    df_["x0"] = 0
    df_["y0"] = 0
    df_["psi"] = 0
    df_["U"] = np.sqrt(df_["u"] ** 2 + df_["v"] ** 2)
    df_["beta"] = -np.arctan2(df_["v"], df_["u"])
    df_["thrust"] = thrust_

    df_["delta"] = 0

    result = vmm.simulator.simulate(
        df_=df_,
        parameters=parameters,
        ship_parameters=ship_parameters,
        control_keys=["delta", "thrust"],
        primed_parameters=True,
        prime_system=ps,
    )

    return result.result.copy()


## Without thrust works fine:
def test_acceleration_no_thrust():

    df_result = simulate(thrust_=0)

    ## Does acceleration and velocity match?
    # integral(u1d,dt) == u ?
    u_ = df_result["u"].values
    u1d_ = df_result["u1d"].values

    t_ = np.array(df_result.index)

    u_integrated = u_[0] + scipy.integrate.cumtrapz(y=u1d_, x=t_)
    u_integrated = np.concatenate([[u_[0]], u_integrated])

    assert_almost_equal(u_, u_integrated, decimal=4)


## With thrust did not work, but this bug has now been corrected:
def test_acceleration_thrust():

    df_result = simulate(thrust_=50)

    ## Does acceleration and velocity match?
    # integral(u1d,dt) == u ?
    u_ = df_result["u"].values
    u1d_ = df_result["u1d"].values

    t_ = np.array(df_result.index)

    u_integrated = u_[0] + scipy.integrate.cumtrapz(y=u1d_, x=t_)
    u_integrated = np.concatenate([[u_[0]], u_integrated])

    # fig,ax=plt.subplots()
    # df_result.plot(y='u', ax=ax)
    # ax.plot(t_, u_integrated, '--', label='u_integrated')
    # ax.legend()
    # plt.show()

    assert_almost_equal(u_, u_integrated, decimal=4)
