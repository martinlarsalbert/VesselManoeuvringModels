import numpy as np
from vessel_manoeuvring_models.kalman_filter import filter_yaw_example
import pandas as pd
import matplotlib.pyplot as plt

A = np.array([[0, 1], [0, -0.1]])

B = np.array([[0], [1]])

E = np.array(
    [[0], [1]],
)

C = np.array([[1, 0]])


def simulate_model(x0, us, ws, A, B, E, t):

    simdata = []
    x = x0
    h = t[1] - t[0]
    for i, u in enumerate(us):

        u = us[i]  # input
        w = ws[i]  # process noise

        x_dot = A @ x + B * u + E * w

        ## Euler integration (k+1)
        x = x + h * x_dot

        simdata.append(x.flatten())

    simdata = np.array(simdata)
    df = pd.DataFrame(simdata, columns=["yaw", "yaw rate"], index=t)
    df["u"] = us
    df["w"] = ws
    return df


def test_simulate():

    # simulation parameters
    N = 1000  # no. of iterations
    f_s = 10  # sampling frequency [Hz]
    h = 1 / f_s  # sampling time: h  = 1/f_s (s)
    t = np.arange(0, N * h, h)

    # initial values for x
    x0 = np.array([[0, 0]]).T
    us = 0.1 * np.sin(0.1 * t)  # inputs
    np.random.seed(42)
    ws = 0.1 * np.random.normal(scale=1, size=N)  # process noise

    df = simulate_model(x0=x0, us=us, ws=ws, A=A, B=B, E=E, t=t)

    # fig, axes = plt.subplots(nrows=3)
    # df.plot(y="u", label="u (input)", ax=axes[0])
    # df.plot(y="yaw", ax=axes[1])
    # df.plot(y="yaw rate", ax=axes[2])
    # plt.show()


def test_filter():

    # simulation parameters
    N = 100  # no. of iterations
    f_m = 1  # yaw angle measurement frequency [Hz]
    h_m = 1 / f_m  # sampling time: h  = 1/f_s (s)
    t = np.arange(0, N * h_m, h_m)

    # initial values for x
    x0 = np.array([[0, 0]]).T
    us = 0.1 * np.sin(0.1 * t)  # inputs
    np.random.seed(42)
    ws = 0.1 * np.random.normal(scale=1, size=N)  # process noise

    df = simulate_model(x0=x0, us=us, ws=ws, A=A, B=B, E=E, t=t)

    ## Measured yaw angle:
    df["epsilon"] = 0.1 * np.random.normal(scale=1, size=N)  # measurement noise
    df["y"] = df["yaw"] + df["epsilon"]
    ys = np.zeros((N, 1))  # 1!
    ys[:, 0] = df["y"].values

    ## Discretisized system matrixes:
    f_s = 10  # sampling frequency [Hz]
    h = 1 / f_s  # sampling time: h  = 1/f_s (s)
    Ad = np.eye(2) + h * A
    Bd = h * B
    Cd = C
    Ed = h * E

    # initialization of Kalman filter
    x0 = np.array([[0, 0]]).T
    P_prd = np.diag([1, 1])
    Qd = 1
    Rd = 10

    filter_yaw_example(
        x0=x0,
        P_prd=P_prd,
        h_m=h_m,
        h=h,
        us=us,
        ys=ys,
        Ad=Ad,
        Bd=Bd,
        Cd=Cd,
        Ed=Ed,
        Qd=Qd,
        Rd=Rd,
    )
