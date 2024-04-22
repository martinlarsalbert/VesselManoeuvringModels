import numpy as np
import pandas as pd
from numpy.linalg.linalg import inv, pinv
import pandas as pd
from scipy.interpolate import interp1d
from vessel_manoeuvring_models.angles import smallest_signed_angle

from dataclasses import dataclass

def is_column_vector(x:np.ndarray):
    return ((x.ndim == 2) and 
            (x.shape[1] == 1))  # Column vector

@dataclass
class FilterResult:
    t : np.ndarray
    x_prd : np.ndarray
    x_hat : np.ndarray
    K : np.ndarray
    epsilon: np.ndarray
    P_hat: np.ndarray
    P_prd: np.ndarray
    y: np.ndarray


class KalmanFilter:

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        E: np.ndarray=None,
        state_columns=["x0", "y0", "psi", "u", "v", "r"],
        measurement_columns=["x0", "y0", "psi"],
        input_columns=["delta"],
        control_columns=[],
        angle_columns=['psi'],
    ) -> pd.DataFrame:
        """Kalman Filter
        
        Parameters
        ----------
        A : np.ndarray [n,n], matrix to form state transition: Phi = I + A*h
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

        """
        self.A=A
        self.B=B
        self.H=H
        self.Q=Q
                            
        self.R=R
        
        self.state_columns = state_columns
        self.input_columns = input_columns
        self.control_columns = control_columns
        self.measurement_columns = measurement_columns
        self.angle_columns = angle_columns
        
        self.n = len(state_columns)          # No. of state vars.
        self.m = len(input_columns)          # No. of input vars.
        self.p = len(measurement_columns)    # No. of measurement vars.
        
        
        assert self.A.shape == (self.n,self.n)
        if self.m > 0:
            assert self.B.shape == (self.n,self.m)
        assert self.H.shape == (self.p,self.n)
        assert self.Q.shape == (self.n,self.n)
        assert self.R.shape == (self.p,self.p)
   
        if E is None:
            self.E = np.eye(self.n)  # The entire Q is used
        else:
            self.E=E

        self.mask_angles = [key in angle_columns for key in measurement_columns]
        
        
    def Phi(self, x_hat: np.ndarray, control: pd.Series, h:float):
        A = self.A
        Phi = np.eye(self.n) + A*h
        
        return Phi
        
    def state_prediction(self,x_hat, Phi, control: pd.Series, h:float):
        x_prd = Phi @ x_hat
        return x_prd
    
    def predict(self, x_hat : np.ndarray, P_hat : np.ndarray, u : np.ndarray, h : float, control: pd.Series=None):
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
        
        B = self.B
        E = self.E
        Q = self.Q
        self.Delta = Delta =  B*h
        self.Gamma = Gamma = E*h
          
        Phi = self.Phi(x_hat=x_hat, control=control, h=h)
        
        # Predictor (k+1)
        # State estimate propagation:
        x_prd = self.state_prediction(x_hat=x_hat, Phi=Phi, control=control, h=h)
        
        if self.m>0:
            # Add inputs if they exist:
            x_prd+=Delta @ u
            
        # Error covariance propagation:
        #P_prd = Phi @ P_hat @ Phi.T + Gamma * Q @ Gamma.T ## Note Q not Qd!
        Qd = Q*h
        #P_prd = Phi @ P_hat @ Phi.T + Qd
        
        #E = np.array(
        #    [
        #        [0, 0, 0],
        #        [0, 0, 0],
        #        [0, 0, 0],
        #        [1, 0, 0],
        #        [0, 1, 0],
        #        [0, 0, 1],
        #    ],
        #)
        #Ed=E*h
        #P_prd = Phi @ P_hat @ Phi.T + Qd*h**2
        P_prd = Phi @ P_hat @ Phi.T + Qd*h**2
        
        
        return x_prd, P_prd
    
    def update(self, y:np.ndarray, P_prd:np.ndarray, x_prd:np.ndarray, h:float, dead_reckoning=False):
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
        
        if dead_reckoning:
            H = 0*self.H
        else:
            H = self.H
        
        R = self.R
        Rd = R*h
        n_states = len(x_prd)
        
        # Compute kalman gain matrix:
        S = H @ P_prd @ H.T + Rd  # System uncertainty
        K = P_prd @ H.T @ pinv(S)

        # Error covariance update:
        IKC = np.eye(n_states) - K @ H        
        #P_hat = IKC * P_prd @ IKC.T + K @ Rd @ K.T
        P_hat = IKC @ P_prd @ IKC.T + K @ Rd @ K.T
        
        epsilon = y - H @ x_prd  # Error between meassurement (y) and predicted measurement H @ x_prd
        
        epsilon_old = epsilon.copy()
        
        epsilon[self.mask_angles] = smallest_signed_angle(
            epsilon[self.mask_angles]
        )  # Smalles signed angle

        
        #if not (epsilon == epsilon_old).all():
        #    a = 1
        
        # State estimate update:
        x_hat = x_prd + K @ epsilon
                        
        return x_hat, P_hat, K, epsilon.flatten()
    
    
    def filter(self,
        data: pd.DataFrame,
        P_0: np.ndarray,
        x0: np.ndarray = None,
        h: float = None,
        )->FilterResult:
        """_summary_

        Args:
            data (pd.DataFrame): Measurement and input data
            P_0 (np.ndarray): Initial covariance
            x0 (np.ndarray, optional): Initial state. If None first row of data is used.
            h (float, optional): _description_. Time step of filter. If None --> timestep filter == timestep data

        Returns:
            FilterResult: _description_
        """

        
        data = data.copy()
        
        assert data.index.name == 'time', "You need to name index 'time' to assert that it is time"
        
        if h is None:
            ts = data.index  # Data and filter have the same time
        else:
            ts = np.arange(data.index[0], data.index[-1] + h, h)  # Data and filter have different times.
        
        time_interpolator = interp1d(x=ts, y=ts, kind='nearest', assume_sorted=True)
        filter_time = time_interpolator(data.index)
        filter_to_measurement_time = pd.Series(index=filter_time, data=data.index)
        mask=filter_to_measurement_time.index.duplicated()
        filter_to_measurement_time=filter_to_measurement_time[~mask].copy()
        
        #data["time_filter"] = pd.Series(data.index).apply(lambda t_measurement: ts[np.argmin(np.abs(ts-t_measurement))])
            
        
        assert set(self.input_columns).issubset(data.columns), "Some inputs missing in data"
        us = data[self.input_columns].values.T
        assert set(self.measurement_columns).issubset(data.columns), "Some measurements missing in data"
        #ys = data[self.measurement_columns].values.T
        
        if x0 is None:
            x0 = data.iloc[0][self.state_columns].values.reshape(self.n,1)
                
        #assert ys.ndim==2
        assert is_column_vector(x0)
        assert x0.shape[0] == self.n, f"x0 should be a column vector with {self.n} elements"
        
        N = len(ts)
        
        if len(us)!=N:
            us = np.tile(us,[1,N])
        
        # Initialize:
        x_prds=np.zeros((self.n,N))
        x_prd = x0
        x_prds[:,0] = x_prd.flatten()
        
        x_hats=np.zeros((self.n,N))
        P_hats=np.zeros((N,self.n,self.n))
        P_prds=np.zeros((N,self.n,self.n))
        Ks=np.zeros((N,self.n,self.p))
        epsilon=np.zeros((self.p,N))
        ys = np.zeros((N,self.p))
        
        P_prd = P_0.copy()

        u = data.iloc[0][self.input_columns].values.reshape((self.m,1))
        control=data.iloc[0][self.control_columns]
        
        x_hat = x0
        P_hat = P_0
        
        for i,t in enumerate(ts):
            
                        
            t = ts[i]
            if i<(N-1):
                h = ts[i+1]-ts[i]
            
            #if self.m > 0:
            #    u = us[:,[i]]
            #else:
            #    u = us
            
            if t in filter_to_measurement_time:
                ## Measurements exist near this time, make an update...            
                
                measurement_time = filter_to_measurement_time[t]
                y = data.loc[measurement_time,self.measurement_columns].values.reshape((self.p,1))
                u = data.loc[measurement_time,self.input_columns].values.reshape((self.m,1))
                control=data.loc[measurement_time,self.control_columns]
                
                dead_reckoning=False
            else:
                dead_reckoning=True
                        
            
            #if i<(N-1):
            #    x_prd,P_prd = self.predict(x_hat=x_hat, P_hat=P_hat, u=u, h=h, control=control)
            #    x_prds[:,i+1] = x_prd.flatten()

            x_hat, P_hat, K, epsilon[:,i] = self.update(y=y, P_prd=P_prd, x_prd=x_prd, h=h, dead_reckoning=dead_reckoning)
            
            x_prd,P_prd = self.predict(x_hat=x_hat, P_hat=P_hat, u=u, h=h, control=control)
            
        
            x_prds[:,i] = x_prd.flatten()
            x_hats[:,i] = x_hat.flatten()
            Ks[i,:,:] = K
            P_hats[i,:,:] = P_hat
            P_prds[i,:,:] = P_prd
            ys[i,:] = y.flatten()
        
        #i+=1
        #x_hat, P_hat, K, epsilon[:,i] = self.update(y=ys[:,[i]], P_prd=P_prd, x_prd=x_prd,h=h)
        #x_hats[:,i] = x_hat.flatten()
        #Ks[i,:,:] = K
        
        result = FilterResult(t=ts, x_prd=x_prds, x_hat=x_hats, K=Ks, epsilon=epsilon, P_hat=P_hats, P_prd=P_prds, y=ys)
        
        return result
        
    
    def simulate(self, x0: np.ndarray, t:np.ndarray, us: np.ndarray, controls: pd.DataFrame=None)->np.ndarray:
        """Simulate with the state transition model

        Args:
            x0 (np.ndarray): Initial state
            t (np.ndarray): Time vector
            us (np.ndarray): Input vector

        Returns:
            np.ndarray: _description_
        """
        
        N = len(t)
        
        P_hat = np.eye(len(x0))        

        if len(us)!=N:
            us = np.tile(us,[1,N])
        
        if controls is None:
            controls = pd.DataFrame(index=t)
        
        x_hats=np.zeros((self.n,N))
        x_hat = x0.copy()
        
        for i,t_ in enumerate(t[0:-1]):
            x_hats[:,i] = x_hat.flatten()
            
            if self.m > 0:
                u = us[:,[i]]
            else:
                u = us
                
            control = controls.loc[t_,self.control_columns]
            
            h = t[i+1]-t[i]
            x_hat,_ = self.predict(x_hat=x_hat, P_hat=P_hat, u=u, h=h, control=control)
            
        x_hats[:,i+1] = x_hat.flatten()
            
        return x_hats
            
            
        