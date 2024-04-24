import numpy as np
import pandas as pd
from vessel_manoeuvring_models.KF_multiple_sensors import KalmanFilter
from typing import AnyStr, Callable
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
from vessel_manoeuvring_models.angles import smallest_signed_angle


class ExtendedKalmanFilter(KalmanFilter):

    def __init__(
        self,
        model: ModularVesselSimulator,
        B: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        E: np.ndarray = None,
        lambda_f: Callable = None,
        lambda_Phi: Callable = None,
        state_columns=["x0", "y0", "psi", "u", "v", "r"],
        measurement_columns=["x0", "y0", "psi"],
        input_columns=["delta"],
        control_columns=["delta"],
        angle_columns=["psi"],
    ) -> pd.DataFrame:
        """_summary_

        Args:
        model (ModularVesselSimulator): the predictor model
        B : np.ndarray [n,m], Control input model
        H : np.ndarray [p,n], Ovservation model
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

        Returns:
            pd.DataFrame: _description_
        """

        self.B = B
        self.H = H
        self.Q = Q

        self.R = R

        self.state_columns = state_columns
        self.input_columns = input_columns
        self.control_columns = control_columns
        self.measurement_columns = measurement_columns

        self.n = len(state_columns)  # No. of state vars.
        self.m = len(input_columns)  # No. of input vars.
        self.p = len(measurement_columns)  # No. of measurement vars.

        if self.m > 0:
            assert self.B.shape == (self.n, self.m)
        assert self.H.shape == (self.p, self.n)
        assert self.Q.shape == (self.n, self.n)
        assert self.R.shape == (self.p, self.p)

        if E is None:
            self.E = np.eye(self.n)  # The entire Q is used
        else:
            self.E = E

        self.lambda_f = lambda_f
        self.lambda_Phi = lambda_Phi
        self.model = model

        self.mask_angles = [key in angle_columns for key in measurement_columns]

    def Phi(self, x_hat: np.ndarray, control: pd.Series, h: float):

        states_dict = pd.Series(index=self.state_columns, data=x_hat.flatten())

        Phi = self.lambda_Phi(
            **states_dict,
            **control,
            **self.model.parameters,
            **self.model.ship_parameters,
            h=h,
        )

        return Phi

    def state_prediction(self, x_hat, Phi, control: pd.Series, h: float):
        states_dict = pd.Series(index=self.state_columns, data=x_hat.flatten())

        f = self.lambda_f(
            **states_dict,
            **control,
            **self.model.parameters,
            **self.model.ship_parameters,
        )

        x_prd = x_hat + f * h

        # x_prd = Phi @ x_hat

        return x_prd


def update_gradient(data):
    data["x1d"] = np.gradient(data["x0"], data.index)
    data["y1d"] = np.gradient(data["y0"], data.index)

    data["u"] = data["x1d"] * np.cos(data["psi"]) + data["y1d"] * np.sin(data["psi"])
    data["v"] = -data["x1d"] * np.sin(data["psi"]) + data["y1d"] * np.cos(data["psi"])
    data["r"] = np.gradient(data["psi"], data.index)

    data["u1d"] = np.gradient(data["u"], data.index)
    data["v1d"] = np.gradient(data["v"], data.index)
    data["r1d"] = np.gradient(data["r"], data.index)
