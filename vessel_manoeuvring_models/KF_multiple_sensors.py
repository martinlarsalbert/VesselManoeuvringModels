import numpy as np
import pandas as pd
from numpy.linalg.linalg import inv, pinv

from dataclasses import dataclass

def is_column_vector(x:np.ndarray):
    return ((x.ndim == 2) and 
            (x.shape[1] == 1))  # Column vector

@dataclass
class FilterResult:
    x_prd : np.ndarray
    x_hat : np.ndarray
    K : np.ndarray
    epsilon: np.ndarray

class KalmanFilter:

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        E: np.ndarray=None,
    ) -> pd.DataFrame:
        """Example kalman filter
        
        Parameters
        ----------
        A : np.ndarray
        B : np.ndarray
        H : np.ndarray
            observation model
        Q : np.ndarray
            process noise
        R : np.ndarray
            measurement noise
        E : np.ndarray
        Returns
        -------
        pd.DataFrame
            data frame with filtered data
        """
        self.A=A
        self.B=B
        self.H=H
        self.Q=Q
                            
        self.R=R
        
        self.n = self.A.shape[0]    # No. of state vars.
        
        if len(self.B) == 0:
            self.m = 0
        else:
            self.m = self.B.shape[1]    # No. of input vars.
            assert self.B.shape[0] == self.n
                
        self.p = self.H.shape[0]    # No. of measurement vars.
        
        assert self.A.shape[1] == self.n
        assert self.H.shape[1] == self.n
        assert self.Q.shape[0] == self.n
        assert self.Q.shape[1] == self.n
        assert self.R.shape[0] == self.p
        assert self.R.shape[1] == self.p
        
        if E is None:
            self.E = np.eye(self.n)  # The entire Q is used
        else:
            self.E=E
        
        

    def predict(self, x_hat, P_hat, u, h):
        
        assert is_column_vector(x_hat)
        
        if self.m > 0:
            assert is_column_vector(u)
        
        A = self.A
        B = self.B
        E = self.E
        Q = self.Q
        self.Delta = Delta =  B*h
        self.Gamma = Gamma = E*h
          
        self.Phi = Phi = np.eye(self.n) + A*h
        #Phi = A
        
        
        # Predictor (k+1)
        # State estimate propagation:
        x_prd = Phi @ x_hat
        if self.m>0:
            # Add inputs if they exist:
            x_prd+=Delta @ u
            
        # Error covariance propagation:
        #P_prd = Phi @ P_hat @ Phi.T + Gamma * Q @ Gamma.T ## Note Q not Qd!
        Qd = Q*h
        P_prd = Phi @ P_hat @ Phi.T + Qd
        
        return x_prd, P_prd
    
    def update(self, y, P_prd, x_prd, h):
            
        H = self.H
        R = self.R
        Rd = R*h
        n_states = len(x_prd)
        
        epsilon = y - H @ x_prd  # Error between meassurement (y) and predicted measurement H @ x_prd
        
        # Compute kalman gain matrix:
        S = H @ P_prd @ H.T + Rd  # System uncertainty
        K = P_prd @ H.T @ inv(S)

        # State estimate update:
        x_hat = x_prd + K @ epsilon
        
        # Error covariance update:
        IKC = np.eye(n_states) - K @ H        
        P_hat = IKC * P_prd @ IKC.T + K @ Rd @ K.T
        
        return x_hat, P_hat, K, epsilon.flatten()
    
    
    def filter(self,
        x0: np.ndarray,
        P_0: np.ndarray,
        #h_m: float,
        h: float,
        us: np.ndarray,
        ys: np.ndarray,):
        """_summary_

        Args:
        x0 : np.ndarray
            initial state [yaw, yaw rate]
        P_prd : np.ndarray
            2x2 array: initial covariance matrix
        h_m : float
            time step measurement [s]
        h : float
            time step filter [s]
        us : np.ndarray
            1D array: inputs
        ys : np.ndarray
            1D array: measured yaw
        """
        
        assert ys.ndim==2
        assert is_column_vector(x0)
        assert x0.shape[0] == self.n, f"x0 should be a column vector with {self.n} elements"
        
        N = ys.shape[1]
        n_measurement_states = ys.shape[0]
        n_states = len(x0)
        
        if len(us)!=N:
            us = np.tile(us,[1,N])
        
        # Initialize:
        x_prds=np.zeros((n_states,N))
        x_prd = x0
        x_prds[:,0] = x_prd.flatten()
        
        x_hats=np.zeros((n_states,N))
        Ks=np.zeros((N,n_states,n_measurement_states))
        epsilon=np.zeros((n_measurement_states,N))
        
        P_prd = P_0.copy() 
        
        for i in range(N-1):
            
            if self.m > 0:
                u = us[:,[i]]
            else:
                u = us
            
            x_hat, P_hat, K, epsilon[:,i] = self.update(y=ys[:,[i]], P_prd=P_prd, x_prd=x_prd, h=h)
            x_hats[:,i] = x_hat.flatten()
            Ks[i,:,:] = K          
            
            x_prd,P_0 = self.predict(x_hat=x_hat, P_hat=P_hat, u=u, h=h)
            x_prds[:,i+1] = x_prd.flatten()
        
        i+=1
        x_hat, P_hat, K, epsilon[:,i] = self.update(y=ys[:,[i]], P_prd=P_prd, x_prd=x_prd,h=h)
        x_hats[:,i] = x_hat.flatten()
        Ks[i,:,:] = K
        
        result = FilterResult(x_prd=x_prds, x_hat=x_hats, K=Ks, epsilon=epsilon)
        #result['x_prd'] = x_prd
        #result['x_hat'] = x_hat
        #result['K'] = K
        
        return result
        
    
    def simulate(self, x0: np.ndarray, t:np.ndarray, us: np.ndarray):
        
        N = len(t)
        
        P_hat = np.eye(len(x0))        

        if len(us)!=N:
            us = np.tile(us,[1,N])
        
        x_hats=np.zeros((self.n,N))
        x_hat = x0.copy()
        
        for i,t_ in enumerate(t[0:-1]):
            x_hats[:,i] = x_hat.flatten()
            
            if self.m > 0:
                u = us[:,[i]]
            else:
                u = us
            
            h = t[i+1]-t[i]
            x_hat,_ = self.predict(x_hat=x_hat, P_hat=P_hat, u=u, h=h)
            
        x_hats[:,i+1] = x_hat.flatten()
            
        return x_hats
            
            
        