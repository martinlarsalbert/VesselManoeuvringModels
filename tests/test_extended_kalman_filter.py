import numpy as np
from vessel_manoeuvring_models.kalman_filter import (
    extended_kalman_filter_example,
    simulate_model,
)

from vessel_manoeuvring_models.extended_kalman_filter import extended_kalman_filter, rts_smoother

import pandas as pd
import matplotlib.pyplot as plt
from sympy import Matrix
from vessel_manoeuvring_models.substitute_dynamic_symbols import lambdify
import sympy as sp

x_1, x_2, a, b, u, w, h = sp.symbols("x_1,x_2, a, b, u, w, h")
jac = sp.eye(2) + Matrix([x_2, a * x_2 * x_2 + b * u + w]).jacobian([x_1, x_2]) * h
lambda_jacobian = lambdify(jac)
f = Matrix([x_2, a * x_2 * sp.Abs(x_2) + b * u + w])
lambda_f = lambdify(f)


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
        x0=x0,
        P_prd=P_prd,
        lambda_f=lambda_f,
        lambda_jacobian=lambda_jacobian,
        h_m=h_m,
        h=h,
        us=us,
        ys=ys,
        Qd=Qd,
        Rd=Rd,
    )


# x_1, x_2, a, b, u, w, h = sp.symbols("x_1,x_2, a, b, u, w, h")
# jac_a = (
#    sp.eye(3) + Matrix([x_2, a * x_2 * x_2 + b * u + w, 0]).jacobian([x_1, x_2, a]) * h
# )
# lambda_jacobian_a = lambdify(jac_a)
# f_a = Matrix([x_2, a * x_2 * sp.Abs(x_2) + b * u + w, 0])
# lambda_f_a = lambdify(f_a)


def lambda_f_constructor(b, w=0):
    def lambda_f_a(x, u):

        f = np.array([[x[1], x[2] * x[1] * np.abs(x[1]) + b * u + w, 0]]).T
        return f

    return lambda_f_a


def lambda_jacobian_constructor(h):
    def lambda_jacobian(x, u):

        jac = np.array(
            [
                [1, h, 0],
                [0, 2 * x[2] * h * np.abs(x[1]) + 1, h * x[1] * np.abs(x[1])],
                [0, 0, 1],
            ]
        )
        return jac

    return lambda_jacobian
