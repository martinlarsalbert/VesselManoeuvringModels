import pytest
from vessel_manoeuvring_models import parameters
from vessel_manoeuvring_models.extended_kalman_vmm import ExtendedKalman
import vessel_manoeuvring_models.models.vmm_martin_simple as vmm

from vessel_manoeuvring_models.parameters import df_parameters
from vessel_manoeuvring_models.substitute_dynamic_symbols import run
from vessel_manoeuvring_models import prime_system
import numpy as np
from vessel_manoeuvring_models.visualization.plot import track_plot
import matplotlib.pyplot as plt
import pandas as pd
import os
import dill


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


@pytest.fixture
def parameters():
    mask = df_parameters["brix_lambda"].notnull()
    df_parameters.loc[mask, "brix_prime"] = df_parameters.loc[mask].apply(
        calculate_prime, ship_parameters=ship_parameters, axis=1
    )

    df_parameters["prime"] = df_parameters["brix_prime"]

    df_parameters.loc["Ydelta", "prime"] = 0.001  # Just guessing
    df_parameters.loc["Ndelta", "prime"] = (
        -df_parameters.loc["Ydelta", "prime"] / 2
    )  # Just guessing

    df_parameters.loc["Nu", "prime"] = 0
    df_parameters.loc["Nur", "prime"] = 0
    #df_parameters.loc["Xdelta", "prime"] = -0.001
    df_parameters.loc["Xr", "prime"] = 0
    df_parameters.loc["Xrr", "prime"] = 0.007
    df_parameters.loc["Xu", "prime"] = -0.001
    df_parameters.loc["Xv", "prime"] = 0
    df_parameters.loc["Xvr", "prime"] = -0.006
    df_parameters.loc["Yu", "prime"] = 0
    df_parameters.loc["Yur", "prime"] = 0.001

    parameters = dict(df_parameters["prime"].copy())
    parameters.pop("Xdelta")
    yield parameters


ps = prime_system.PrimeSystem(**ship_parameters)  # model
ship_parameters_prime = ps.prime(ship_parameters)


@pytest.fixture
def data():
    N_ = 4000
    u = np.deg2rad(
        30
        * np.concatenate(
            (
                -1 * np.ones(int(N_ / 4)),
                1 * np.ones(int(N_ / 4)),
                -1 * np.ones(int(N_ / 4)),
                1 * np.ones(int(N_ / 4)),
            )
        )
    )

    t = np.linspace(0, 50, N_)
    data_ = pd.DataFrame(index=t)
    data_["delta"] = u
    data_["thrust"] = 0.1
    data_["x0"] = 0
    data_["y0"] = 0
    data_["psi"] = 0
    data_["u"] = 3
    data_["v"] = 0
    data_["r"] = 0

    yield data_


def test_filter(data, parameters):

    ## Filter
    ek = ExtendedKalman(vmm=vmm, parameters=parameters, ship_parameters=ship_parameters)

    ## Simulate
    process_noise_u = 0.01
    process_noise_v = 0.01
    process_noise_r = np.deg2rad(0.01)

    N_ = len(data)
    ws = np.zeros((N_, 3))
    ws[:, 0] = np.random.normal(loc=process_noise_u, size=N_)
    ws[:, 1] = np.random.normal(loc=process_noise_v, size=N_)
    ws[:, 2] = np.random.normal(loc=process_noise_r, size=N_)

    E = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
    )

    df_sim = ek.simulate(data=data, ws=ws, E=E, input_columns=["delta","thrust"], solver="Radau")

    ## Measure
    df_measure = df_sim.copy()
    measurement_noise_psi_max = 3
    measurement_noise_psi = np.deg2rad(measurement_noise_psi_max / 3)
    epsilon_psi = np.random.normal(scale=measurement_noise_psi, size=N_)

    measurement_noise_xy_max = 2
    measurement_noise_xy = measurement_noise_xy_max / 3
    epsilon_x0 = np.random.normal(scale=measurement_noise_xy, size=N_)
    epsilon_y0 = np.random.normal(scale=measurement_noise_xy, size=N_)

    df_measure["psi"] = df_sim["psi"] + epsilon_psi
    df_measure["x0"] = df_sim["x0"] + epsilon_x0
    df_measure["y0"] = df_sim["y0"] + epsilon_y0

    P_prd = np.diag([0.1, 0.1, np.deg2rad(0.01), 0.001, 0.001, np.deg2rad(0.001)])
    Qd = np.diag([0.01, 0.01, np.deg2rad(0.1)])  # process variances: u,v,r
    t = df_sim.index
    h = t[1] - t[0]
    Rd = h * np.diag(
        [
            measurement_noise_xy ** 2,
            measurement_noise_xy ** 2,
            measurement_noise_psi ** 2,
        ]
    )  # measurement variances: x0,y0,psi

    Cd = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ]
    )

    time_stamps = ek.filter(data=df_measure, input_columns=["delta","thrust"], P_prd=P_prd, Qd=Qd, Rd=Rd, E=E, Cd=Cd)


def test_save_load(data, parameters, tmpdir):

    ## Filter
    ek = ExtendedKalman(vmm=vmm, parameters=parameters, ship_parameters=ship_parameters)
    path = os.path.join(str(tmpdir), "ek.pkl")

    # for key, value in ek.__dict__.items():
    #    try:
    #        dill.dumps(value)
    #    except Exception as e:
    #        print(key)

    ek.save(path)
    ek2 = ExtendedKalman.load(path)

    E = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
    )

    ## Simulate
    process_noise_u = 0.01
    process_noise_v = 0.01
    process_noise_r = np.deg2rad(0.01)
    N_ = len(data)
    ws = np.zeros((N_, 3))
    ws[:, 0] = np.random.normal(loc=process_noise_u, size=N_)
    ws[:, 1] = np.random.normal(loc=process_noise_v, size=N_)
    ws[:, 2] = np.random.normal(loc=process_noise_r, size=N_)

    df_sim = ek2.simulate(data=data, ws=ws, E=E, input_columns=["delta","thrust"])
