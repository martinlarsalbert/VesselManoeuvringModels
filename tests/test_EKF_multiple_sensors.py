from vessel_manoeuvring_models.EKF_multiple_sensors import ExtendedKalmanFilter
import numpy as np
import pytest
import pandas as pd
from dataclasses import dataclass, field
import sympy as sp
from sympy import symbols, Eq, ImmutableDenseMatrix
from vessel_manoeuvring_models.substitute_dynamic_symbols import (
    lambdify,
    run,
    expression_to_python_method,
)

g = 9.81


@dataclass
class Model:
    parameters: dict = field(default_factory=dict)
    ship_parameters: dict = field(default_factory=dict)


@pytest.fixture
def ekf():

    x, x1d, g = symbols("x,\dot{x},g")
    states = [x, x1d, g]
    f_ = ImmutableDenseMatrix([x1d, -g, 0])
    jac = f_.jacobian(states)
    h = symbols("h")  # Time step
    Phi = sp.eye(len(states), len(states)) + jac * h

    subs_simpler = [
        (x1d, "x1d"),
    ]
    lambda_f = expression_to_python_method(
        expression=f_.subs(subs_simpler),
        function_name="lambda_f",
        substitute_functions=False,
    )

    lambda_Phi = expression_to_python_method(
        expression=Phi.subs(subs_simpler),
        function_name="lambda_jacobian",
        substitute_functions=False,
    )

    dummy_model = Model()

    B = np.array(
        [[]]
    )  # Discrete time input transition matrix # Discrete time transition matrix

    var_x = 2
    var_x1d = 0.001
    var_x_Q = 0.1
    var_x1d_Q = 0.3
    Q = np.diag([0, var_x1d_Q**2, 0])  # Covariance matrix of the process model

    H = np.array(
        [
            [1, 0, 0],
        ]
    )  # Measurement transition matrix
    R = np.diag([var_x**2])  # Covariance matrix of the measurement

    ekf = ExtendedKalmanFilter(
        model=dummy_model,
        B=B,
        H=H,
        Q=Q,
        R=R,
        lambda_f=lambda_f,
        lambda_Phi=lambda_Phi,
        state_columns=["x", "x1d", "g"],
        measurement_columns=["x"],
        input_columns=[],
        control_columns=[],
        angle_columns=[],
    )

    yield ekf


@pytest.fixture
def data(ekf):
    g = 9.81
    dt = 0.1
    x0 = np.array([[0, 0, g]]).T
    dt = 0.1
    t = np.arange(0, 5, dt)

    u = np.array([[]])
    x_true = ekf.simulate(
        x0=x0,
        t=t,
        us=u,
    )

    np.random.seed(42)
    var_x = 2
    epsilon = np.random.normal(scale=var_x**2, size=len(t))
    ys = x_true.copy()
    ys[0, :] += epsilon
    ys = np.array(ys[0:1, :])

    data = pd.DataFrame(index=t)
    data.index.name = "time"

    data[ekf.measurement_columns] = ys.T

    return data


def test_filter(ekf, data):
    x0 = np.array([[0, 0, g]]).T
    P_0 = np.diag([0.1, 0.01, 0])
    u = np.array([[]]).T
    result = ekf.filter(data=data, P_0=P_0, x0=x0)
