import numpy as np
import pandas as pd
from vessel_manoeuvring_models.extended_kalman_filter import extended_kalman_filter, simulate
import matplotlib.pyplot as plt
import pytest


def lambda_f_constructor(K, T_1):
    def lambda_f(x, input):
        delta = input["delta"]
        f = np.array([[x[1], (K * delta - x[1]) / T_1]]).T
        return f

    return lambda_f


def lambda_f_constructor2(K):
    def lambda_f(x, input):
        delta = input["delta"]
        T_1 = x[2]  # Note! T_1 is the third state now!
        r = x[1]

        f = np.array([[r, (K * delta - r) / T_1, 0]]).T
        return f

    return lambda_f


def do_simulation(h_, lambda_f, N_=4000):

    t_ = np.arange(0, N_ * h_, h_)

    us = np.deg2rad(
        np.concatenate(
            (
                -10 * np.ones(int(N_ / 4)),
                10 * np.ones(int(N_ / 4)),
                -10 * np.ones(int(N_ / 4)),
                10 * np.ones(int(N_ / 4)),
            )
        )
    )
    data = pd.DataFrame(index=t_)
    data["delta"] = us

    np.random.seed(42)
    E = np.array([[0, 1]]).T
    process_noise = np.deg2rad(0.01)
    ws = process_noise * np.random.normal(size=N_)

    data["psi"] = 0
    data["r"] = 0
    df = simulate(data=data, lambda_f=lambda_f, E=E, ws=ws, state_columns=["psi", "r"])

    measurement_noise = np.deg2rad(0.5)
    df["epsilon"] = measurement_noise * np.random.normal(size=N_)
    df["psi_measure"] = df["psi"] + df["epsilon"]
    df["psi_deg"] = np.rad2deg(df["psi"])
    df["psi_measure_deg"] = np.rad2deg(df["psi_measure"])
    df["delta_deg"] = np.rad2deg(df["delta"])

    return df


@pytest.mark.skip("Fix this some day")
def test_simulate():

    N_ = 4000
    T_1_ = 1.8962353076056344
    K_ = 0.17950970687951323
    h_ = 0.02
    lambda_f = lambda_f_constructor(K=K_, T_1=T_1_)
    do_simulation(h_=h_, lambda_f=lambda_f, N_=N_)


def lambda_f_constructor2(K):
    def lambda_f(x, input):
        delta = input["delta"]
        T_1 = x[2]  # Note! T_1 is the third state now!
        r = x[1]

        f = np.array([[r, (K * delta - r) / T_1, 0]]).T
        f = f.reshape(x.shape)

        return f

    return lambda_f


def lambda_jacobian_constructor(h, K):
    def lambda_jacobian(x, input):

        T_1 = x[2]  # Note! T_1 is the third state now!
        delta = input["delta"]
        r = x[1]

        jac = np.array(
            [
                [1, h, 0],
                [0, 1 - h / T_1, -h * (K * delta - r) / T_1 ** 2],
                [0, 0, 1],
            ]
        )
        return jac

    return lambda_jacobian


@pytest.mark.skip("Fix this some day")
def test_filter():

    N_ = 8000
    T_1_ = 1.8962353076056344
    K_ = 0.17950970687951323
    h_ = 0.01

    n_states = 3
    n_measurements = 1
    n_hidden = n_states - n_measurements

    lambda_f = lambda_f_constructor(K=K_, T_1=T_1_)
    df = do_simulation(h_=h_, lambda_f=lambda_f, N_=N_)

    ## Filter:
    lambda_jacobian = lambda_jacobian_constructor(h=h_, K=K_)
    lambda_f2 = lambda_f_constructor2(K=K_)

    x0 = np.deg2rad([0, 0, 0.1]).reshape(n_states, 1)
    P_prd = np.diag([np.deg2rad(1), np.deg2rad(0.1), 0.1])

    Qd = np.deg2rad(np.diag([0, 0.5]))

    Rd = np.diag([np.deg2rad(1)])

    data = df[["delta"]]
    data["psi"] = df["psi_measure"]

    E_ = np.array(
        [[0, 0], [1, 0], [0, 1]],
    )

    C_ = np.array([[1, 0, 0]])

    Cd_ = C_

    time_steps = extended_kalman_filter(
        x0=x0,
        P_prd=P_prd,
        lambda_f=lambda_f2,
        lambda_jacobian=lambda_jacobian,
        E=E_,
        Qd=Qd,
        Rd=Rd,
        Cd=Cd_,
        state_columns=["psi", "r", "T_1"],
        measurement_columns=["psi"],
        data=data,
    )

    ## Post process:
    x_hats = np.array([time_step["x_hat"].flatten() for time_step in time_steps]).T
    time = np.array([time_step["time"] for time_step in time_steps]).T
    Ks = np.array([time_step["K"] for time_step in time_steps]).T

    ## Plotting:
    n = len(P_prd)
    fig, axes = plt.subplots(nrows=n)

    keys = ["psi", "r"]
    for i, key in enumerate(keys):

        ax = axes[i]
        df.plot(y=key, ax=ax, label="True")
        if key == "psi":
            df.plot(y="psi_measure", ax=ax, label="Measured", zorder=-1)

        ax.plot(time, x_hats[i, :], "-", label="kalman")
        ax.set_ylabel(key)
        ax.legend()

    fig.show()

    dummy = 1
