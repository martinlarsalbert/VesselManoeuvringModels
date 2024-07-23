"""UKF implementation according to: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/10-Unscented-Kalman-Filter.ipynb
"""

import numpy as np
import scipy
from numpy import array, zeros
from numpy.linalg import inv

class SigmaPoints():
    
    def __init__(self, n, alpha=0.001, beta=2, kappa=0):
        
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        self._num_sigmas = 2*n + 1
        
        self.calculate_weights()
    
    def num_sigmas(self):
        return self._num_sigmas
        
    def calculate_weights(self):
        
        n = self.n
        kappa = self.kappa
        alpha = self.alpha
        beta = self.beta
        
        ## Weights:
        lambda_ = alpha**2 * (n + kappa) - n
        self.lambda_ = lambda_
        self.Wc = np.full(2*n + 1,  1. / (2*(n + lambda_)))
        self.Wm = np.full(2*n + 1,  1. / (2*(n + lambda_)))
        self.Wc[0] = lambda_ / (n + lambda_) + (1. - alpha**2 + beta)
        self.Wm[0] = lambda_ / (n + lambda_)
            
    
    def sigma_points(self, x,P):

        n = self.n
        
        x = array(x)
        P = array(P)
                        
        ## Sigma points:
        sigmas = np.zeros((2*n+1, n))
        U = scipy.linalg.cholesky((n+self.lambda_)*P) # sqrt

        sigmas[0] = x
        for k in range (n):
            sigmas[k+1]   = x + U[k]
            sigmas[n+k+1] = x - U[k]
                
        return sigmas
    



def unscented_transform(transformed_sigma_points, Wm, Wc):
    
    x = np.dot(Wm, transformed_sigma_points)
    
    kmax, n = transformed_sigma_points.shape
    P = zeros((n, n))
    for k in range(kmax):
        y = transformed_sigma_points[k] - x
        P += Wc[k] * np.outer(y, y) 
    
    return x,P

class SigmaPointKalmanFilter():
    
    def __init__(
        self,
        fx,
        hx,
        Q: np.ndarray,
        R: np.ndarray,
        sigma_points: SigmaPoints,
        dt: float,
        state_columns=["x0", "y0", "psi", "u", "v", "r"],
        measurement_columns=["x0", "y0", "psi"],
        input_columns=["delta"],
        control_columns=["delta"],
        angle_columns=["psi"],
        kind='UKF',
    ):
        self.fx = fx
        self.hx = hx
        self.Q = Q
        self.R = R
        self.dt = dt

        self.state_columns = state_columns
        self.input_columns = input_columns
        self.control_columns = control_columns
        self.measurement_columns = measurement_columns
        self.angle_columns = angle_columns

        self.n = len(state_columns)  # No. of state vars.
        self.m = len(input_columns)  # No. of input vars.
        self.p = len(measurement_columns)  # No. of measurement vars.
        self.sigma_points = sigma_points
        self._num_sigmas = self.sigma_points.num_sigmas()
        
        self.x = zeros(self.n)
        self.P = np.diag(np.ones(self.n))
        #self.xp = self.x.copy()
        #self.Pp = self.P.copy()
        
    
    def predict(self, x=None, P=None):
        """ Performs the predict step of the UKF. On return, 
        self.xp and self.Pp contain the predicted state (xp) 
        and covariance (Pp). 'p' stands for prediction.
        """
    
        if x is None:
            x = self.x
        
        if P is None:
            P = self.P
    
        # calculate sigma points for given mean and covariance
        sigmas = self.sigma_points.sigma_points(x, P)

        self.sigmas_f = zeros((self.sigma_points._num_sigmas,self.n))
        
        for i in range(self.sigma_points._num_sigmas):
            self.sigmas_f[i] = self.fx(sigmas[i], self.dt)
    
        self.xp, self.Pp = unscented_transform(
                       self.sigmas_f, self.sigma_points.Wm, self.sigma_points.Wc)
        self.Pp+=self.Q
    
    
    def update(self, z):

        self.sigmas_h = zeros((self.sigma_points._num_sigmas,self.p))
        
        # rename for readability
        sigmas_f = self.sigmas_f
        sigmas_h = self.sigmas_h

        # transform sigma points into measurement space
        for i in range(self._num_sigmas):
            sigmas_h[i] = self.hx(sigmas_f[i])

        # mean and covariance of prediction passed through UT
        zp, Pz = unscented_transform(sigmas_h, self.sigma_points.Wm, self.sigma_points.Wc)
        Pz+=self.R

        # compute cross variance of the state and the measurements
        Pxz = np.zeros((self.n, self.p))
        for i in range(self._num_sigmas):
            Pxz += self.sigma_points.Wc[i] * np.outer(sigmas_f[i] - self.xp,
                                        sigmas_h[i] - zp)

        K = np.dot(Pxz, inv(Pz)) # Kalman gain

        self.x = self.xp + np.dot(K, z - zp)
        self.P = self.Pp - np.dot(K, Pz).dot(K.T)