import numpy as np
import pandas as pd
from vessel_manoeuvring_models.KF_multiple_sensors import KalmanFilter, FilterResult, is_column_vector
from typing import AnyStr, Callable
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
from vessel_manoeuvring_models.angles import smallest_signed_angle
from numpy.linalg.linalg import inv, pinv
from math import factorial
from vessel_manoeuvring_models.sigma_points import sigma_points
from numpy import zeros
from filterpy.kalman.sigma_points import MerweScaledSigmaPoints

class SigmaPointKalmanFilter(KalmanFilter):
    
    def __init__(
        self,
        model: ModularVesselSimulator,
        B: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        lambda_f: Callable = None,
        state_columns=["x0", "y0", "psi", "u", "v", "r"],
        measurement_columns=["x0", "y0", "psi"],
        input_columns=["delta"],
        control_columns=["delta"],
        angle_columns=["psi"],
        kind='UKF',
    ):
        """_summary_

        Args:
        model (ModularVesselSimulator): the predictor model
        B : np.ndarray [n,m] or lambda function!, Control input model
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
        state_columns (list, optional): _description_. Defaults to ["x0", "y0", "psi", "u", "v", "r"].
        measurement_columns (list, optional): _description_. Defaults to ["x0", "y0", "psi"].
        input_columns (list, optional): _description_. Defaults to ["delta"].
        angle_columns (list, optional): the angle states are treated with "smallest angle" in the epsilon calculation.
        kind (str, optional): 'UKF', or 'CKF'. Defaults to 'UKF'.
        """
        
        self.B = B
        self.H = H
        self.Q = Q

        self.R = R

        self.state_columns = state_columns
        self.input_columns = input_columns
        self.control_columns = control_columns
        self.measurement_columns = measurement_columns
        self.angle_columns = angle_columns

        self.n = len(state_columns)  # No. of state vars.
        self.m = len(input_columns)  # No. of input vars.
        self.p = len(measurement_columns)  # No. of measurement vars.
        
        if self.m > 0:
            if not callable(self.B):
                assert self.B.shape == (self.n, self.m), f"n:{self.n}, m:{self.m}"
                
        if not callable(self.H):
            assert self.H.shape == (self.p, self.n), f"p:{self.p}, n:{self.n}"

        assert self.Q.shape == (self.n, self.n), f"n:{self.n}, n:{self.n}"
        assert self.R.shape == (self.p, self.p), f"p:{self.p}, p:{self.p}"

        self.lambda_f = lambda_f
        self.model = model

        self.mask_angles = [key in angle_columns for key in measurement_columns]
        self.kind = kind
        
        n_ = 2
        kappa = 3-self.n
        self.sigma_points = MerweScaledSigmaPoints(self.n, alpha=.01, beta=2., kappa=kappa)
        
    def predict(
        self,
        x_hat: np.ndarray,
        P_hat: np.ndarray,
        u: np.ndarray,
        h: float,
        control: pd.Series = None,
    ):
        """Make a predicton with the state transition model

        Args:
            x_hat (np.ndarray): _description_
            P_hat (np.ndarray): _description_
            u (np.ndarray): _description_
            h (float): _description_

        Returns:
            x_prd: predicted state
            P_prd: error covariance propagation
        """

        assert is_column_vector(x_hat)

        if self.m > 0:
            assert is_column_vector(u)

        
        SP,W = sigma_points(x=x_hat, P=P_hat, kind=self.kind)
        #SP = self.sigma_points.sigma_points(x=x_hat.flatten(), P=P_hat).T
        #W = self.sigma_points.Wm
        
        
        n,N = SP.shape
        g = zeros((n,N))
        x=zeros(n)
        P=zeros((n,n))
        
        for i in range(N):
            sigma_point = SP[:,i]
            
            g_ = self.state_prediction(x_hat=sigma_point, control=control, u=u, h=h)
            g[:,i] = g_.flatten()
            
            x = x + W[i]*g[:,i]
        
        for i in range(N):
            P = P + W[i]*(g[:,i]-x) @ (g[:,i]-x).T
        Qd = self.Q * h
        P = P + Qd   

        return x.reshape(self.n,1), P
    
    def state_prediction(self, x_hat, control: pd.Series, u:np.ndarray, h: float):
        
        x_hat = x_hat.reshape(self.n,1)
        
        states_dict = pd.Series(index=self.state_columns, data=x_hat.flatten())
        input_dict = pd.Series(index=self.input_columns, data=u.flatten())
        
        if 'u' in states_dict and 'v' in states_dict:
            states_dict['U'] = np.sqrt(states_dict['u']**2 + states_dict['v']**2)
            
        try:
            calculation = self.model.calculate_forces(
                states_dict=states_dict[self.model.states_str], control=control[self.model.control_keys]
            )
        except:
            calculation = {}
            
        f = self.lambda_f(
            **states_dict,
            **input_dict,
            **control,
            **self.model.parameters,
            **self.model.ship_parameters,
            **calculation,
            h=h,
        )

        x_prd = x_hat + f * h

        return x_prd
    
    def h(self, x_hat: np.ndarray, control: pd.Series, h: float) -> np.ndarray:
        """Measurement/observation model

        Args:
            x_hat (np.ndarray): _description_
            h (float): _description_

        Returns:
            np.ndarray: _description_
        """
        return self.H @ x_hat

    def update(
        self,
        y: np.ndarray,
        P_prd: np.ndarray,
        x_prd: np.ndarray,
        x_hat: np.ndarray,
        h: float,
        control: pd.Series = None,
        dead_reckoning=False,
    ):
        """Update prediction with measurements.

        Args:
            y (np.ndarray): _description_
            P_prd (np.ndarray): _description_
            x_prd (np.ndarray): _description_
            h (float): _description_
            dead_reckoning (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """

        Rd = self.R * h
        n_states = len(x_prd)

        SP,W = sigma_points(x=x_prd, P=P_prd, kind=self.kind)
        n,N = SP.shape
        
        m = len(y)
        g = zeros((m,N))
        
        # y_k:
        y_pred =zeros((m,1))
        for i in range(N):
            sigma_point = SP[:,i].reshape(self.n,1)
            g_ = self.h(sigma_point, control=control, h=h)
            y_pred = y_pred + W[i]*g_
            g[:,i] = g_.flatten()
            
                
        # P_xy:
        P_xy = zeros((n,m))
        for i in range(N):
            sigma_point = SP[:,i].reshape(self.n,1)
            g_ = g[:,i].reshape(m,1)
            P_xy = P_xy + W[i]*(sigma_point - x_prd) @ (g_ - y_pred).T
            
        # S_k:
        S_k = zeros((m,m))
        for i in range(N):
            g_ = g[:,i].reshape(m,1)
            S_k = S_k + W[i]*(g_ - y) @ (g_ - y_pred).T
        S_k = S_k + Rd
        
        x = x_prd + P_xy @ inv(S_k) @ (y-y_pred)
        P = P_prd - P_xy @ inv(S_k) @ P_xy.T
        
        #K = P_prd @ self.H.T @ pinv(S_k)
        K = None
        
        v = (
            y - y_pred
        )  # Innovation: difference between prediction and measurement
        
        v[self.mask_angles] = smallest_signed_angle(
            v[self.mask_angles]
        )  # Smalles signed angle

        return x, P, K, v.flatten()