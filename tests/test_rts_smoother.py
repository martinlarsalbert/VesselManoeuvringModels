import numpy as np
import pandas as pd
from vessel_manoeuvring_models.extended_kalman_filter import extended_kalman_filter, rts_smoother
import matplotlib.pyplot as plt

process_noise = np.deg2rad(0.1 / 3)
measurement_noise = np.deg2rad(3.0 / 3)


def lambda_f_constructor(K, T_1):
    def lambda_f(x, input):
        delta = input["delta"]
        f = np.array([[x[1], (K * delta - x[1]) / T_1]]).T
        return f

    return lambda_f


def simulate(E, ws, t, inputs, lambda_f, h_):

    simdata = []
    x_ = np.deg2rad(np.array([[0, 0]]).T)

    for i in range(len(inputs)):

        input = inputs.iloc[i]
        w_ = ws[i]
        x_ = x_ + h_ * lambda_f(x=x_.flatten(), input=input) + E * w_

        simdata.append(x_.flatten())

    simdata = np.array(simdata)
    df = pd.DataFrame(simdata, columns=["psi", "r"], index=t)
    df["delta"] = inputs["delta"]

    return df


def do_simulation(K, T_1, h_, lambda_f, N_=4000):

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
    inputs = pd.DataFrame(index=t_)
    inputs["delta"] = us

    np.random.seed(42)
    E = np.array([[0, 1]]).T

    ws = np.random.normal(scale=process_noise, size=N_)
    df = simulate(E=E, ws=ws, t=t_, inputs=inputs, lambda_f=lambda_f, h_=h_)

    df["epsilon"] = np.random.normal(scale=measurement_noise, size=N_)
    df["psi_measure"] = df["psi"] + df["epsilon"]
    df["psi_deg"] = np.rad2deg(df["psi"])
    df["psi_measure_deg"] = np.rad2deg(df["psi_measure"])
    df["delta_deg"] = np.rad2deg(df["delta"])

    return df


def test_simulate():

    N_ = 4000
    T_1_ = 1.8962353076056344
    K_ = 0.17950970687951323
    h_ = 0.02
    lambda_f = lambda_f_constructor(K=K_, T_1=T_1_)
    do_simulation(K=K_, T_1=T_1_, h_=h_, lambda_f=lambda_f, N_=N_)


def lambda_jacobian_constructor(h, K, T_1):
    def lambda_jacobian(x, input):

        r = x[1]

        jac = np.array(
            [
                [1, h],
                [0, 1 - h / T_1],
            ]
        )
        return jac

    return lambda_jacobian


def test_rts_smoother():

    N_ = 500
    T_1_ = 1.8962353076056344
    K_ = 0.17950970687951323
    h_ = 0.4

    n_states = 2
    n_measurements = 1

    lambda_f = lambda_f_constructor(K=K_, T_1=T_1_)
    df = do_simulation(K=K_, T_1=T_1_, h_=h_, lambda_f=lambda_f, N_=N_)

    ## Kalman Filter:
    lambda_jacobian = lambda_jacobian_constructor(h=h_, K=K_, T_1=T_1_)
    lambda_f2 = lambda_f_constructor(K=K_, T_1=T_1_)

    x0 = np.deg2rad([5, 0]).reshape(n_states, 1)
    P_prd = np.diag([np.deg2rad(10), np.deg2rad(1)])

    Qd = np.deg2rad(np.diag([process_noise ** 2]))

    Rd = np.diag([measurement_noise ** 2])

    ys = df["psi_measure"].values
    data = df[["delta"]]
    data["psi"] = df["psi_measure"]

    E_ = np.array([[0, 1]]).T

    Cd_ = Cd = np.array([[1, 0]])

    b = 1
    Bd = np.array([[0, K_ / T_1_]]).T * h_

    time_steps = extended_kalman_filter(
        x0=x0,
        P_prd=P_prd,
        lambda_f=lambda_f2,
        lambda_jacobian=lambda_jacobian,
        data=data,
        E=E_,
        Qd=Qd,
        Rd=Rd,
        Cd=Cd_,
        state_columns=["psi", "r"],
        measurement_columns=["psi"],
    )

    ## Post process Kalman filter:
    x_hats = np.array([time_step["x_hat"].flatten() for time_step in time_steps]).T
    time = np.array([time_step["time"] for time_step in time_steps]).T
    df_kalman = pd.DataFrame(data=x_hats.T, index=time, columns=["psi", "r"])
    df_kalman["delta"] = df["delta"]

    smooth_time_steps = rts_smoother(
        time_steps=time_steps,
        lambda_jacobian=lambda_jacobian,
        Qd=Qd,
        lambda_f=lambda_f2,
        E=E_,
    )

    ## Post process rts smoother:
    x_hats = np.array(
        [time_step["x_hat"].flatten() for time_step in smooth_time_steps]
    ).T
    time = np.array([time_step["time"] for time_step in smooth_time_steps]).T
    df_rts = pd.DataFrame(data=x_hats.T, index=time, columns=["psi", "r"])
    df_rts["delta"] = df["delta"]

    ## Plotting:
    n = len(P_prd)
    fig, axes = plt.subplots(nrows=n)

    keys = ["psi", "r"]
    for i, key in enumerate(keys):

        ax = axes[i]
        df.plot(y=key, ax=ax, label="True")
        if key == "psi":
            df.plot(y="psi_measure", ax=ax, label="Measured", zorder=-1)

        df_kalman.plot(y=key, label="kalman", ax=ax)
        df_rts.plot(y=key, label="RTS", ax=ax)

        ax.set_ylabel(key)
        ax.legend()

    fig.show()

    dummy = 1
