import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from vessel_manoeuvring_models.kalman_filter import filter_yaw, rts_smoother


def simulate(Ad, Bd, E, ws, t, us):

    simdata = []
    x_ = np.deg2rad(np.array([[0, 0]]).T)
    for u_, w_ in zip(us, ws):

        x_ = (Ad @ x_ + Bd * u_) + E * w_

        simdata.append(x_.flatten())

    simdata = np.array(simdata)
    df = pd.DataFrame(simdata, columns=["psi", "r"], index=t)
    df["delta"] = us

    return df


def do_simulation(h, Ad, Bd):
    ## Simulate
    N = 4000
    t_ = np.arange(0, N * h, h)

    us = np.deg2rad(
        np.concatenate(
            (
                -10 * np.ones(int(N / 4)),
                10 * np.ones(int(N / 4)),
                -10 * np.ones(int(N / 4)),
                10 * np.ones(int(N / 4)),
            )
        )
    )

    E = np.array([[0, 1]]).T
    process_noise = np.deg2rad(0.01)
    ws = process_noise * np.random.normal(size=N)
    df = simulate(Ad=Ad, Bd=Bd, E=E, ws=ws, t=t_, us=us)

    measurement_noise = np.deg2rad(3)
    df["epsilon"] = measurement_noise * np.random.normal(size=N)
    df["psi_measure"] = df["psi"] + df["epsilon"]
    df["psi_deg"] = np.rad2deg(df["psi"])
    df["psi_measure_deg"] = np.rad2deg(df["psi_measure"])
    df["delta_deg"] = np.rad2deg(df["delta"])
    return df


def test_filter_yaw():

    np.random.seed(42)

    T_1 = 1.8962353076056344
    K = 0.17950970687951323
    h = 0.02

    Ad = np.array([[1, h], [0, 1 - h / T_1]])
    Bd = np.array([[0, -K * h / T_1]]).T

    df = do_simulation(h=h, Ad=Ad, Bd=Bd)

    ## Filter:
    x0 = np.deg2rad(np.array([[0, 0]]).T)
    P_prd = np.diag(np.deg2rad([1, 0.1]))

    Qd = np.deg2rad(np.diag([0, 0.5]))

    Rd = np.deg2rad(1)

    ys = df["psi_measure"].values
    us = df["delta"].values

    E_ = np.array(
        [[0, 0], [0, 1]],
    )

    C_ = np.array([[1, 0]])

    Cd_ = C_
    Ed_ = h * E_

    time_steps = filter_yaw(
        x0=x0,
        P_prd=P_prd,
        h_m=h,
        h=h,
        us=us,
        ys=ys,
        Ad=Ad,
        Bd=Bd,
        Cd=Cd_,
        Ed=Ed_,
        Qd=Qd,
        Rd=Rd,
    )
    x_hats = np.array([time_step["x_hat"] for time_step in time_steps]).T
    time = np.array([time_step["time"] for time_step in time_steps]).T
    Ks = np.array([time_step["K"] for time_step in time_steps]).T

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


def test_rts_smoother():

    np.random.seed(42)

    T_1 = 1.8962353076056344
    K = 0.17950970687951323
    h = 0.02

    Ad = np.array([[1, h], [0, 1 - h / T_1]])
    Bd = np.array([[0, -K * h / T_1]]).T

    df = do_simulation(h=h, Ad=Ad, Bd=Bd)

    ## Filter:
    x0 = np.deg2rad(np.array([[0, 0]]).T)
    P_prd = np.diag(np.deg2rad([1, 0.1]))

    Qd = np.deg2rad(np.diag([0, 2]))

    Rd = np.deg2rad(0.1)

    ys = df["psi_measure"].values
    us = df["delta"].values

    E_ = np.array(
        [[0, 0], [0, 1]],
    )

    C_ = np.array([[1, 0]])

    Cd_ = C_
    Ed_ = h * E_

    time_steps = filter_yaw(
        x0=x0,
        P_prd=P_prd,
        h_m=h,
        h=h,
        us=us,
        ys=ys,
        Ad=Ad,
        Bd=Bd,
        Cd=Cd_,
        Ed=Ed_,
        Qd=Qd,
        Rd=Rd,
    )
    x_hats = np.array([time_step["x_hat"] for time_step in time_steps])
    P_hats = [time_step["P_hat"] for time_step in time_steps]
    time = np.array([time_step["time"] for time_step in time_steps]).T

    x, P, K, Pp = rts_smoother(x_hats=x_hats, P_hats=P_hats, Ad=Ad, Bd=Bd, Qd=Qd, us=us)

    n = len(P_prd)
    fig, axes = plt.subplots(nrows=n)

    keys = ["psi", "r"]
    for i, key in enumerate(keys):

        ax = axes[i]
        df.plot(y=key, ax=ax, label="True")
        if key == "psi":
            df.plot(y="psi_measure", ax=ax, label="Measured", zorder=-1)

        ax.plot(time, x_hats[:, i], "-", label="kalman")
        ax.plot(time, x[:, i], "-", label="rts smoother")
        ax.set_ylabel(key)
        ax.legend()

    fig.show()
    dummy = 1
