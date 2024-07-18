import numpy as np
import pandas as pd
from vessel_manoeuvring_models.KF_multiple_sensors import KalmanFilter, FilterResult
from typing import AnyStr, Callable
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
from vessel_manoeuvring_models.angles import smallest_signed_angle
from numpy.linalg.linalg import inv, pinv
from math import factorial

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

        if E is None:
            self.E = np.eye(self.n)  # The entire Q is used
        else:
            self.E = E

        self.lambda_f = lambda_f
        self.lambda_Phi = lambda_Phi
        self.model = model

        self.mask_angles = [key in angle_columns for key in measurement_columns]
       

    def Phi(self, x_hat: np.ndarray, control: pd.Series, u:np.ndarray, h: float):

        states_dict = pd.Series(index=self.state_columns, data=x_hat.flatten())
        input_dict = pd.Series(index=self.input_columns, data=u.flatten())

        if 'u' in states_dict and 'v' in states_dict:
            states_dict['U'] = np.sqrt(states_dict['u']**2 + states_dict['v']**2)
        
        Phi = self.lambda_Phi(
            **states_dict,
            **input_dict,
            **control,
            **self.model.parameters,
            **self.model.ship_parameters,
            h=h,
        )

        return Phi

    def state_prediction(self, x_hat, control: pd.Series, u:np.ndarray, h: float):
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
        #n = 1
        #x_prd = x_hat 
        #for i in range(1,n+1):
        #    x_prd+= (f**i * h**i)/factorial(i) 

        # x_prd = Phi @ x_hat

        return x_prd
    
    def control_prediction(self,x_hat, control: pd.Series, u:np.ndarray, h:float):
        
        if callable(self.B):
            states_dict = pd.Series(index=self.state_columns, data=x_hat.flatten())
            input_dict = pd.Series(index=self.input_columns, data=u.flatten())
            b = self.B(
                **states_dict,
                **input_dict,
                **control,
                **self.model.parameters,
                **self.model.ship_parameters,
                h=h,)
            return b*h
            
        else:
            B = self.B
            self.Delta = Delta = B * h
            return Delta @ u

    def H_k(self, x_hat: np.ndarray, control: pd.Series, h: float) -> np.ndarray:
        """Linear observation model

        Args:
            x_hat (np.ndarray): _description_
            h (float): _description_

        Returns:
            np.ndarray: _description_
        """

        if callable(self.H):
            states_dict = pd.Series(index=self.state_columns, data=x_hat.flatten())
            return self.H(
                **states_dict,
                **control,
                **self.model.parameters,
                **self.model.ship_parameters,
                h=h,
            )
        else:
            return self.H
        
    def smoother(self, results: FilterResult)->FilterResult:
        """RTS smoother 
        
        Args:
            results (FilterResult): _description_

        Returns:
            FilterResult: _description_
        """
        
        n = len(results.t)

        new_results = results.copy()
       
        for k in range(n - 2, -1, -1):
            
            h = new_results.t[k+1]-new_results.t[k]
            control = pd.Series(new_results.control[k,:], index=new_results.control_columns)
            u = new_results.u[k,:].reshape((self.m, 1))
            x_hat = new_results.x_hat[:,k].reshape(self.n,1)
            
            Phi = self.Phi(x_hat=x_hat, control=control, u=u, h=h)
            P_hat = new_results.P_hat[k,:,:]
                        
            Qd = self.Q * h
            Pp = Phi @ P_hat @ Phi.T + Qd # predicted covariance

            K = P_hat @ Phi.T @ pinv(Pp)

            
            #f_hat = self.lambda_f(x=x_hat.flatten(), input=input).reshape((self.n, 1))
            #x_prd = x_hat + h * f_hat
            x_prd = self.state_prediction(x_hat=x_hat, control=control, u=u, h=h)
            
            
            new_results.x_prd[:, k] = x_prd.flatten()

            x_hat_future = new_results.x_hat[:,k + 1].reshape(self.n,1)
            new_results.x_hat[:,k]+= (K @ (x_hat_future - x_prd)).flatten()
            new_results.P_hat[k,:,:]+= K @ (new_results.P_hat[k + 1,:,:] - Pp) @ K.T
            #new_results.K[k,:,0:3] = K[:,0:3]
            #new_results.K[k,:,3:] = K[:,6:]
            

        return new_results


class ExtendedKalmanFilterVMM(ExtendedKalmanFilter):

    def __init__(
        self,
        model: ModularVesselSimulator,
        B: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        E: np.ndarray = None,
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
        state_columns (list, optional): _description_. Defaults to ["x0", "y0", "psi", "u", "v", "r"].
        measurement_columns (list, optional): _description_. Defaults to ["x0", "y0", "psi"].
        input_columns (list, optional): _description_. Defaults to ["delta"].
        angle_columns (list, optional): the angle states are treated with "smallest angle" in the epsilon calculation.

        Returns:
            pd.DataFrame: _description_
        """

        super().__init__(
            model=model,
            B=B,
            H=H,
            Q=Q,
            R=R,
            E=E,
            state_columns=state_columns,
            measurement_columns=measurement_columns,
            input_columns=input_columns,
            control_columns=control_columns,
            angle_columns=angle_columns,
            lambda_f=self.lambda_f,
            lambda_Phi=self.lambda_Phi,
        )

    def lambda_f(self, **kwargs) -> np.ndarray:

        kwargs = pd.Series(kwargs)

        states_dict = kwargs[
            [
                "x0",
                "y0",
                "psi",
                "u",
                "v",
                "r",
            ]
        ]

        control = kwargs[self.control_columns]
        calculation = self.model.calculate_forces(
            states_dict=states_dict, control=control
        )

        result = self.model.lambda_f(
            **states_dict,
            **control,
            **self.model.parameters,
            **self.model.ship_parameters,
            **calculation,
            h=kwargs["h"],
        )

        return result

    # def lambda_Phi(self, x, input: pd.Series) -> np.ndarray:
    def lambda_Phi(self, **kwargs) -> np.ndarray:

        kwargs = pd.Series(kwargs)

        states_dict = kwargs[
            [
                "x0",
                "y0",
                "psi",
                "u",
                "v",
                "r",
            ]
        ]

        control = kwargs[self.control_columns]

        return self.model.calculate_jacobian(
            states_dict=states_dict, control=control, h=kwargs["h"]
        )


def update_gradient(data):
    data["x1d"] = np.gradient(data["x0"], data.index)
    data["y1d"] = np.gradient(data["y0"], data.index)

    data["u"] = data["x1d"] * np.cos(data["psi"]) + data["y1d"] * np.sin(data["psi"])
    data["v"] = -data["x1d"] * np.sin(data["psi"]) + data["y1d"] * np.cos(data["psi"])
    data["r"] = np.gradient(data["psi"], data.index)

    data["u1d"] = np.gradient(data["u"], data.index)
    data["v1d"] = np.gradient(data["v"], data.index)
    data["r1d"] = np.gradient(data["r"], data.index)
