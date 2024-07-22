import numpy as np
import pandas as pd
from vessel_manoeuvring_models.KF_multiple_sensors import FilterResult, is_column_vector
from typing import AnyStr, Callable
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
from vessel_manoeuvring_models.angles import smallest_signed_angle
from numpy.linalg.linalg import inv, pinv
from math import factorial
from vessel_manoeuvring_models.sigma_points import sigma_points
from numpy import zeros
from vessel_manoeuvring_models.UKF import SigmaPointKalmanFilter
from numpy import sin,cos
from scipy.stats import multivariate_normal
from numpy import zeros
from filterpy.kalman.sigma_points import MerweScaledSigmaPoints


class DualBering(SigmaPointKalmanFilter):
    
    def __init__(
        self,
        Q: np.ndarray,
        R: np.ndarray,
        s1: np.ndarray,
        s2: np.ndarray,
        state_columns=["px","py","v","phi","omega"],
        measurement_columns=["phi1","phi2"],  # bearings
        input_columns=[],
        control_columns=[],
        angle_columns=[],
        kind='UKF',
    ):
        """_summary_

        Args:
            Q (np.ndarray): _description_
            R (np.ndarray): _description_
            s1 (numpy.ndarray): Sensor position (2D) for sensor 1.
            s2 (numpy.ndarray): Sensor position (2D) for sensor 2.
            state_columns (list, optional): _description_. Defaults to ["px","py","v","phi","omega"].
            measurement_columns (list, optional): _description_. Defaults to ["phi1","phi2"].
            control_columns (list, optional): _description_. Defaults to [].
            angle_columns (list, optional): _description_. Defaults to [].
            kind (str, optional): _description_. Defaults to 'UKF'.
        """
        

        self.Q = Q
        self.R = R

        self.state_columns = state_columns
        self.input_columns = input_columns
        self.control_columns = control_columns
        self.measurement_columns = measurement_columns
        self.angle_columns = angle_columns

        self.n = len(state_columns)  # No. of state vaqs.
        self.m = len(input_columns)  # No. of input vaqs.
        self.p = len(measurement_columns)  # No. of measurement vaqs.
        self.kind = kind
        
        self.s1=s1
        self.s2=s2
        
        n_ = 2
        kappa = 3-self.n
        self.sigma_points = MerweScaledSigmaPoints(self.n, alpha=.01, beta=2., kappa=kappa)
        
        self.mask_angles = [key in angle_columns for key in measurement_columns]
    
    def state_prediction(self, x_hat, control: pd.Series, u:np.ndarray, h: float):
        """_summary_

        COORDINATEDTURNMOTION calculates the predicted state using a coordinated
        turn motion model, and also calculated the motion model Jacobian
        
        Input:
           x           [5 x 1] state vector
           T           [1 x 1] Sampling time
        
        Output:
           fx          [5 x 1] motion model evaluated at state x
           Fx          [5 x 5] motion model Jacobian evaluated at state x
        
         NOTE: the motion model assumes that the state vector x consist of the
         following states:
           px          X-position
           py          Y-position
           v           velocity
           phi         heading
           omega       turn-rate
        """
        T = h
        x_hat = x_hat.reshape(self.n,1)
        x = x_hat
        

        px = x[0]
        py = x[1]
        v = x[2]
        phi = x[3]
        omega = x[4]

        fx = np.array([
            px + T*v*cos(phi),
            py + T*v*sin(phi),
            v,
            phi + T*omega,
            omega,
        ]).reshape(5,1)
        
        return fx
    
    def genNonLinearStateSequence(self,x_0, Q, N, h, qs=None, UKF=False):
        """GENLINEAqsTATESEQUENCE generates an N+1-long sequence of states using a 
            Gaussian prior and a linear Gaussian process model
        
        Input:
           x_0         [n x 1] Prior mean
           Q           [n x n] Process noise covariance
           N           [1 x 1] Number of states to generate
        
        Output:
           X           [n x N+1] State vector sequence
        
        """
        

        n_states = len(x_0)
        X = zeros((n_states,N+1))

        mu_q = zeros(n_states)
        
        q = multivariate_normal(mean=mu_q.flatten(), cov=Q, allow_singular=True)
        
        def get_q(k):
            if qs is None:
                return q.rvs()
            else:
                return qs[:,k]
            
        X[:,0] = x_0.flatten() + get_q(0)

        for k in range(1,N+1):
            x_hat = X[:,k-1].reshape(n_states,1)
            
            if UKF:
                P_fake = np.diag(np.ones(self.n))/1000
                x_prd,_ = self.predict(x_hat=x_hat, P_hat=P_fake, u=None, h=h, control=None)
            else:
                x_prd = self.state_prediction(x_hat=x_hat, control=None, u=None, h=h)
            
            X[:,k] = (x_prd.flatten() + get_q(k)).flatten()

        #if UKF:
        #    X[0:2,:]*=1.63

        return X


    def dual_bearing_measurement(self, x):
        """
        DualBearingMeasurement calculates the bearings from two sensors, located in
        s1 and s2, to the position given by the state vector x. Also returns the
        Jacobian of the model at x.

        Args:
        x (numpy.ndarray): State vector, the first two elements are 2D position.
        s1 (numpy.ndarray): Sensor position (2D) for sensor 1.
        s2 (numpy.ndarray): Sensor position (2D) for sensor 2.

        Returns:
        numpy.ndarray: Measurement vector.
        numpy.ndarray: Measurement model Jacobian.
        """
        
        s1 = self.s1
        s2 = self.s2
        
        n = len(x)
        hx = np.array([
            np.arctan2(x[1] - s1[1], x[0] - s1[0]),
            np.arctan2(x[1] - s2[1], x[0] - s2[0])
        ])

        #x_k = x[0] - s1[0]
        #y_k = x[1] - s1[1]
        #ddx1 = -y_k / (x_k**2 + y_k**2)
        #ddy1 = x_k / (x_k**2 + y_k**2)
#
        #x_k = x[0] - s2[0]
        #y_k = x[1] - s2[1]
        #ddx2 = -y_k / (x_k**2 + y_k**2)
        #ddy2 = x_k / (x_k**2 + y_k**2)
#
        #Hx = np.zeros((2, n))
        #Hx[0, 0] = ddx1
        #Hx[0, 1] = ddy1
        #Hx[1, 0] = ddx2
        #Hx[1, 1] = ddy2

        #return hx, Hx
        return hx
    
    def dual_bearing_measurement_jacobian(self, x):
        s1 = self.s1
        s2 = self.s2
        
        n = len(x)

        x_k = x[0] - s1[0]
        y_k = x[1] - s1[1]
        ddx1 = -y_k / (x_k**2 + y_k**2)
        ddy1 = x_k / (x_k**2 + y_k**2)

        x_k = x[0] - s2[0]
        y_k = x[1] - s2[1]
        ddx2 = -y_k / (x_k**2 + y_k**2)
        ddy2 = x_k / (x_k**2 + y_k**2)

        Hx = np.zeros((2, n))
        Hx[0, 0] = ddx1
        Hx[0, 1] = ddy1
        Hx[1, 0] = ddx2
        Hx[1, 1] = ddy2

        return Hx
    
    
    def h(self, x_hat: np.ndarray, control: pd.Series, h: float) -> np.ndarray:
        """Measurement/observation model

        Args:
            x_hat (np.ndarray): _description_
            h (float): _description_

        Returns:
            np.ndarray: _description_
        """
        return self.dual_bearing_measurement(x=x_hat)
    
    def gen_non_linear_measurement_sequence(self,X, h, R):
        """
        GenNonLinearMeasurementSequence generates observations of the states
        sequence X using a non-linear measurement model.

        Args:
        X (numpy.ndarray): State vector sequence of shape [n x N+1].
        h (function): Measurement model function handle.
                      Takes as input x (state) and returns hx and Hx,
                      measurement model and Jacobian evaluated at x.
        R (numpy.ndarray): Measurement noise covariance matrix of shape [m x m].

        Returns:
        numpy.ndarray: Measurement sequence of shape [m x N].
        """
        n, N_ = X.shape
        N = N_ - 1
        m, n_ = R.shape

        Y = np.zeros((m, N))
        mu_r = np.zeros(m)

        for k in range(N):
            r = np.random.multivariate_normal(mu_r, R)
            x = X[:, k+1]
            hx = self.dual_bearing_measurement(x=x)
            Y[:, k] = hx.flatten() + r

        return Y

    