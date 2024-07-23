"""UKF implementation according to: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/10-Unscented-Kalman-Filter.ipynb
"""

import numpy as np
import scipy
from numpy import array, zeros
from numpy.linalg import inv
import pandas as pd
from vessel_manoeuvring_models.KF_multiple_sensors import FilterResult, is_column_vector
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from vessel_manoeuvring_models.angles import smallest_signed_angle


class FilterResultUKF(FilterResult):
    
    def __init__(self,n,m,p,N:int,filter):
        
        self.state_columns=filter.state_columns
        self.measurement_columns=filter.measurement_columns
        self.input_columns=filter.input_columns
        self.control_columns=filter.control_columns
        self.angle_columns=filter.angle_columns
        
        self.N = N
        self.t = zeros(N)
        self.x_prd=zeros((N,n))
        self.x_hat=zeros((N,n))
        self.P_hat=zeros((N,n,n))
        self.P_prd=zeros((N,n,n))
        self.K = zeros((N,n,n))
        self.v = zeros((N,p))
        self.y = zeros((N,p))
        self.control = zeros((N,len(self.control_columns)))
               
                
        
    def append_state(self,filter,k:int, t:float):
        
        self.t[k] = t
        self.x_prd[k] = filter.xp
        self.x_hat[k] = filter.x
        self.P_hat = filter.P
        self.P_prd = filter.Pp
        self.y[k] = filter.z
        self.v[k] = filter.v
        self.control[k] = filter.control.values
   
    @property
    def df(self):

        df_states = pd.DataFrame(
            data=self.x_hat, index=self.t, columns=self.state_columns
        )

        df_control = pd.DataFrame(
            data=self.control, index=self.t, columns=self.control_columns
        )

        df = pd.concat((df_states, df_control), axis=1)

        return df
    
    def copy(self):
        return deepcopy(self)
    
    def save(self, path: str):
        """Save model to pickle file

        Parameters
        ----------
        path : str
            Ex:'model.pkl'
        """

        with open(path, mode="wb") as file:
            dill.dump(self, file=file, recurse=True)
        
    @classmethod
    def load(cls, path: str):
        """Load model from pickle file

        Parameters
        ----------
        path : str
            Ex:'model.pkl'
        """

        with open(path, mode="rb") as file:
            obj = dill.load(file=file)

        return obj
    
    @property
    def innovation(self):
        df_states = pd.DataFrame(
            data=self.x_hat, index=self.t, columns=self.state_columns
        )
        df_measurements = pd.DataFrame(self.y, index=self.t, columns=self.measurement_columns)
        
        df_innovation = df_measurements - df_states[self.measurement_columns]
        return df_innovation
    
    def plot_innovation(self, type='plot', fig=None, **plot_kwargs):
        """_summary_

        Args:
            type (str, optional): 'plot', 'hist', or 'autocorr'. Defaults to 'plot'.
            fig (_type_, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            : _description_
        """
        
        df_innovation = self.innovation.copy()
        n_inovations = len(df_innovation.columns)
        if fig is None:
            if n_inovations < 4:
                fig,axes=plt.subplots(nrows=n_inovations, ncols=1)
                if n_inovations==1:
                    axes = np.array(axes)
            else:
                fig,axes=plt.subplots(nrows=3, ncols=2)
        else:
            axes = np.array(fig.axes)
             
        
                
        for ax,key in zip(axes.flatten(),df_innovation.columns):
            
            if type=='plot':
                df_innovation[key].plot(ax=ax, **plot_kwargs)
            elif type=='hist':
                if not 'bins' in plot_kwargs:
                    plot_kwargs['bins']=100
                    
                df_innovation[key].hist(ax=ax, **plot_kwargs)
            
            elif type=='autocorr':
                plot_acf(x=df_innovation[key].values, lags=10, ax=ax, zero=False,**plot_kwargs)
            else:
                raise ValueError(f"Bad plot type:{type}")
                        
            ax.set_title(key)

        return fig
    
    def sigmaEllipse2D(self, i, x='x0',y='y0', level=3, npoints=32)->np.ndarray:
        """
        SIGMAELLIPSE2D generates x,y-points which lie on the ellipse describing
        a sigma level in the Gaussian density defined by mean and covariance.
        
        Args:
            mu (np.ndarray): [2 x 1] Mean of the Gaussian density
            Sigma (np.ndarray): [2 x 2] Covariance matrix of the Gaussian density
            level (int, optional): _description_. Which sigma level curve to plot. Can take any positive value, 
                   but common choices are 1, 2 or 3. Default = 3.
            npoints (int, optional): _description_. Number of points on the ellipse to generate. Default = 32.

        Returns:
            np.ndarray: [2 x npoints] matrix. First row holds x-coordinates, second
                   row holds the y-coordinates. First and last columns should 
                   be the same point, to create a closed curve.
        """
        
        i_x = self.state_columns.index(x)
        i_y = self.state_columns.index(y)
                
        mu = np.array([self.x_hat[i_x,i], self.x_hat[i_y,i]]).reshape((2,1))
        Sigma = np.array([[self.P_hat[i,i_x,i_x],self.P_hat[i,i_x,i_y]],
                          [self.P_hat[i,i_y,i_x],self.P_hat[i,i_y,i_y]],  
                         ])
                        
        xy = sigmaEllipse2D(mu=mu, Sigma=Sigma, level=level, npoints=npoints)
        
        return xy
    

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
        self.mask_angles = [key in angle_columns for key in measurement_columns]
        
    
    def predict(self, control:pd.Series, x=None, P=None):
        """ Performs the predict step of the UKF. On return, 
        self.xp and self.Pp contain the predicted state (xp) 
        and covariance (Pp). 'p' stands for prediction.
        """

        self.predicted = True
        
        if x is None:
            x = self.x
        
        if P is None:
            P = self.P
    
        # calculate sigma points for given mean and covariance
        sigmas = self.sigma_points.sigma_points(x, P)

        self.sigmas_f = zeros((self.sigma_points._num_sigmas,self.n))
        
        self.control = control
        for i in range(self.sigma_points._num_sigmas):
            self.sigmas_f[i] = self.fx(sigmas[i], self.dt, control=control)
    
        self.xp, self.Pp = unscented_transform(
                       self.sigmas_f, self.sigma_points.Wm, self.sigma_points.Wc)
        self.Pp+=self.Q
    
    
    def update(self, z):

        self.sigmas_h = zeros((self.sigma_points._num_sigmas,self.p))
        
        if not self.predicted:
            # the update step is run before the first prediction, so that sigma_f needs to be calculated:
            # calculate sigma points for given mean and covariance
            self.sigmas_f = self.sigma_points.sigma_points(self.x, self.P)  # This is just the sigma points, if has not been run through the predictor function fx
            self.xp=self.x
            self.Pp=self.P
                        
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

        self.K = K = np.dot(Pxz, inv(Pz)) # Kalman gain

        self.z = z
        v = z - zp  # innovation
        v[self.mask_angles] = smallest_signed_angle(
            v[self.mask_angles]
        )  # Smalles signed angle
        self.v = v
        
        self.x = self.xp + np.dot(K, v)
        self.P = self.Pp - np.dot(K, Pz).dot(K.T)
        
    def filter(
        self,
        data: pd.DataFrame,
        P_0: np.ndarray,
        x_0: np.ndarray = None,
        dt: float = None,
    ) -> FilterResult:
        """_summary_

        Args:
            data (pd.DataFrame): Measurement and input data
            P_0 (np.ndarray): Initial covariance
            x_0 (np.ndarray, optional): Initial state. If None first row of data is used.
            h (float, optional): _description_. Time step of filter. If None --> timestep filter == timestep data

        Returns:
            FilterResult: _description_
        """
        
        self.P = P_0
        
        if x_0 is None:
            self.x = data.iloc[0]
        else:
            self.x = x_0
        
        ## The filter can run with another time than the data...
        self.interpolating = False
        if dt is None:
            # (filter time == data time)
            assert data.index.name == 'time', "data.index.name must be 'time'"
            
            dts = pd.Series(data.index).diff().dropna().round(decimals=10).unique()
            assert len(dts)==1, f"The time step of the data is not consistant: {dts}. Please specify a dt!"
            
            ts = data.index
            self.dt = dts[0]
        else:
            # (filter time != data time)
            self.interpolating = True
            self.dt = dt
            ts = np.arange(data.index[0],data.index[-1], dt)
            time_interpolator = interp1d(x=ts, y=ts, kind="nearest", assume_sorted=True, bounds_error=False)
            filter_time = time_interpolator(data.index)
            filter_to_measurement_time = pd.Series(index=filter_time, data=data.index)
            mask = filter_to_measurement_time.index.duplicated()
            filter_to_measurement_time = filter_to_measurement_time[~mask].copy()   
        
        result = FilterResultUKF(n=self.n, m=self.m, p=self.p, N=len(ts), filter=self)    
        
        measurement_time = ts[0]
        self.predicted = False
        for k,t in enumerate(ts):
            
            ## Update?
            update=False
            if self.interpolating:
                if t in filter_to_measurement_time:
                    ## Measurements exist near this time, make an update...
                    measurement_time = filter_to_measurement_time[t]
                    update=True
            else:
                measurement_time = t
                update=True
            
            if update:
                row = data.loc[measurement_time]
                z = row[self.measurement_columns].values
                
                self.update(z=z)
            
            control = data.loc[measurement_time,self.control_columns]
            
            ## Predict:
            self.predict(control=control)
                        
            
            result.append_state(filter=self, k=k,t=t)
            
        return result
        
        
            
    def simulate(
        self,
        x_0: np.ndarray,
        N=100,
        dt=1,
        us: np.ndarray=[],
        controls: pd.DataFrame = None,
    ) -> np.ndarray:
        """Simulate initial value problem (IVP) with the state transition model

        Args:
            x0 (np.ndarray): Initial state
            t (np.ndarray): Time vector
            us (np.ndarray): Input vector

        Returns:
            np.ndarray: _description_
        """
  
        if controls is None:
            ts = np.arange(0,N*dt,dt)
        else:
            assert controls.index.name == 'time', "controls.index.name must be 'time'"
            ts = controls.index
            N = len(controls)
  
        x = array(x_0)
        xs = zeros((N,self.n))
        xs[0] = x
            
        for k in range(1,N):
            t = ts[k]
            control = controls.loc[t]
            dt = ts[k] - ts[k-1]
            xs[k] = self.fx(xs[k-1], dt=dt, control=control)
            
        result = pd.DataFrame(xs, columns=self.state_columns, index=ts)
        result.index.name = 'time'
        
        return result
            