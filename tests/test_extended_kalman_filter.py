import numpy as np
from src.kalman_filter import extended_kalman_filter_example, simulate_model
import pandas as pd
import matplotlib.pyplot as plt


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

    df = simulate_model(x0=x0, us=us, ws=ws, t=t)

    # fig, axes = plt.subplots(nrows=3)
    # df.plot(y="u", label="u (input)", ax=axes[0])
    # df.plot(y="x_1", ax=axes[1])
    # df.plot(y="x_2", ax=axes[2])
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

    df = simulate_model(x0=x0, us=us, ws=ws, t=t)

    ## Measured yaw angle:
    df["epsilon"] = 0.1 * np.random.normal(scale=1, size=N)  # measurement noise
    df["y"] = df["x_1"] + df["epsilon"]
    ys = np.zeros((N, 1))  # 1!
    ys[:, 0] = df["y"].values

    ## Discretisized system matrixes:
    f_s = 10  # sampling frequency [Hz]
    h = 1 / f_s  # sampling time: h  = 1/f_s (s)

    # initialization of Kalman filter
    x0 = np.array([[0, 0]]).T
    P_prd = np.diag([1, 1])
    Qd = 1
    Rd = 10

    extended_kalman_filter_example(
        x0=x0, P_prd=P_prd, h_m=h_m, h=h, us=us, ys=ys, Qd=Qd, Rd=Rd
    )
