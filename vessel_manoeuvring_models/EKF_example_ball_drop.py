from vessel_manoeuvring_models.EKF_multiple_sensors import ExtendedKalmanFilter
import numpy as np
import pandas as pd
import sympy as sp
from sympy import ImmutableDenseMatrix, symbols, Eq
from vessel_manoeuvring_models.substitute_dynamic_symbols import (
    lambdify,
    run,
    expression_to_python_method,
)

from dataclasses import dataclass, field
import matplotlib.pyplot as plt


@dataclass
class Model:
    parameters: dict = field(default_factory=dict)
    ship_parameters: dict = field(default_factory=dict)


class ExtendedKalmanFilterExample(ExtendedKalmanFilter):

    def __init__(
        self,
        H: np.ndarray = None,
        Q: np.ndarray = None,
        R: np.ndarray = None,
        measurement_columns=["x"],
        var_x=2,
        var_x1d_Q=0.3,
        noise_amplification=1,
    ):
        """_summary_

        Args:
        model (ModularVesselSimulator): the predictor model
        B : np.ndarray [n,m], Control input model
        H : np.ndarray [p,n] or lambda function!, Ovservation model
            observation model
        Q : np.ndarray [n,n]
            process noise
        R : np.ndarray [p,p]
            measurement noise
        E : np.ndarray
        state_columns : list
            name of state columns
        measurement_columns : list
            name of measurement columns
        input_columns : list
            name of input (control) columns
        lambda_f (Callable, optional): _description_. Defaults to None.
        lambda_Phi (Callable, optional): _description_. Defaults to None.
        state_columns (list, optional): _description_. Defaults to ["x0", "y0", "psi", "u", "v", "r"].
        measurement_columns (list, optional): _description_. Defaults to ["x0", "y0", "psi"].
        input_columns (list, optional): _description_. Defaults to ["delta"].
        angle_columns (list, optional): the angle states are treated with "smallest angle" in the epsilon calculation.


        """

        self.create_predictor_and_transition_matrix()
        model = Model()

        state_columns = ["x", "x1d", "g"]
        measurement_columns = ["x"]
        input_columns = []
        control_columns = []
        angle_columns = []

        self.noise_amplification = noise_amplification

        B = np.array(
            [[]]
        )  # Discrete time input transition matrix # Discrete time transition matrix

        if Q is None:
            Q = np.diag([0, var_x1d_Q**2, 0])  # Covariance matrix of the process model

        if H is None:
            H = np.array(
                [
                    [1, 0, 0],
                ]
            )  # Measurement transition matrix

        if R is None:
            R = self.noise_amplification * np.diag(
                [var_x**2]
            )  # Covariance matrix of the measurement

        super().__init__(
            model=model,
            B=B,
            H=H,
            Q=Q,
            R=R,
            state_columns=state_columns,
            measurement_columns=measurement_columns,
            input_columns=input_columns,
            control_columns=control_columns,
            angle_columns=angle_columns,
            lambda_f=self.lambda_f,
            lambda_Phi=self.lambda_Phi,
        )

    def create_predictor_and_transition_matrix(self):

        x, x1d, g = symbols("x,\dot{x},g")
        states = [x, x1d, g]
        f_ = ImmutableDenseMatrix([x1d, -g, 0])

        jac = f_.jacobian(states)
        h = symbols("h")  # Time step
        Phi = sp.eye(len(states), len(states)) + jac * h

        subs_simpler = [
            (x1d, "x1d"),
        ]

        self.lambda_f = expression_to_python_method(
            expression=f_.subs(subs_simpler),
            function_name="lambda_f",
            substitute_functions=False,
        )

        self.lambda_Phi = expression_to_python_method(
            expression=Phi.subs(subs_simpler),
            function_name="lambda_jacobian",
            substitute_functions=False,
        )

    def generate_data(
        self, t: np.ndarray = None, dt=0.1, g=9.81, var_x=None, random_seed=42
    ):

        self.g = g

        x0 = np.array([[0, 0, g]]).T

        if t is None:
            t = np.arange(0, 5, dt)

        u = np.array([[]])
        x_true = self.simulate(
            x0=x0,
            t=t,
            us=u,
        )

        if not random_seed is None:
            np.random.seed(random_seed)

        if var_x is None:
            var_x = self.noise_amplification * 2

        epsilon = np.random.normal(scale=var_x**2, size=len(t))
        ys = x_true.copy()
        ys[0, :] += epsilon
        ys = np.array(ys[0:1, :])

        data = pd.DataFrame(index=t)
        data.index.name = "time"

        data[self.measurement_columns] = ys.T

        data_true = pd.DataFrame(index=t)
        data_true.index.name = "time"

        data_true[self.state_columns] = x_true.T

        return data, data_true

    def filter(
        self,
        data: pd.DataFrame,
        P_0: np.ndarray = None,
        x0: np.ndarray = None,
        h: float = None,
    ):

        if hasattr(self, "g"):
            g = self.g
        else:
            g = 9.81

        if x0 is None:
            x0 = np.array([[0, 0, g]]).T

        if P_0 is None:
            P_0 = self.noise_amplification * np.diag([0.1, 0.01, 0])

        u = np.array([[]]).T

        return super().filter(data=data, P_0=P_0, x0=x0, h=h)

    def plot(self, data, result, data_true):

        fig, axes = plt.subplots(nrows=2)

        ax = axes[0]
        ax.plot(data_true.index, data_true["x"], "k", label="True")

        ax.plot(data.index, data["x"], "b-", label="Measured")
        ax.plot(result.t, result.x_hat[0, :], "r--", label="Filtered")
        # ax.plot(result.t,result.x_prd[0,:],'m-', label='Predictor');

        ax2 = ax.twinx()
        ax2.plot(result.t, result.K[:, 0, :], ".g-", label="Kalman gain")
        ax2.set_ylabel("Kalman gain", color="g")
        ax2.tick_params(axis="y", colors="g")

        ax.set_ylabel(r"$x$")
        ax.set_xticklabels([])
        ax.legend()

        ax = axes[1]
        ax.plot(data_true.index, data_true["x1d"], "k", label="True")
        ax.plot(result.t, result.x_hat[1, :], "r-")
        ax.set_ylabel(r"$\dot{x}$")
        ax.set_xlabel("Time [s]")
