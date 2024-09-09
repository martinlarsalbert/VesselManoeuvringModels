import pandas as pd
import numpy as np
import sympy as sp
from sympy.matrices.immutable import ImmutableDenseMatrix
from sympy import Eq
import sympy.physics.mechanics as me

from vessel_manoeuvring_models.EKF_multiple_sensors import ExtendedKalmanFilter
from vessel_manoeuvring_models import reference_frames
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
from vessel_manoeuvring_models.substitute_dynamic_symbols import lambdify, run, expression_to_python_method
from vessel_manoeuvring_models.models.modular_simulator import subs_simpler
from vessel_manoeuvring_models.substitute_dynamic_symbols import eq_dottify
from vessel_manoeuvring_models.symbols import *
import vessel_manoeuvring_models.accelerometers6 as accelerometers6

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
        X = sp.MutableDenseMatrix([x_0,y_0,psi,u,v,r,u1d,v1d,r1d]),  # state vector,
        dynamic_symbols = [u,v,r,delta],
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

        self.define_prediction_model(model, X, dynamic_symbols)
    
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
    
    def define_prediction_model(self,model: ModularVesselSimulator, X, dynamic_symbols):
        
        eq_acceleration = model.expand_subsystemequations(model.acceleartion_eq_SI, prime=False)
        eq_acceleration = sp.simplify(eq_acceleration.subs(U,sp.sqrt(u**2+v**2)))
        
        self.lambda_acceleration = expression_to_python_method(eq_dottify(eq_acceleration.subs(subs_simpler),), function_name="acceleration", )
                
        
        ## defining the transition model:
        #x_dot = ImmutableDenseMatrix([u1d,v1d,r1d])
        
        dynamic_symbols = [u,v,r]
        dynamic_symbols_subs={symbol: me.dynamicsymbols(symbol.name) for symbol in dynamic_symbols} 

        subs={
            dynamic_symbols_subs[u].diff():u1d,
            dynamic_symbols_subs[v].diff():v1d,
            dynamic_symbols_subs[r].diff():r1d,

            dynamic_symbols_subs[u]:u,
            dynamic_symbols_subs[v]:v,
            dynamic_symbols_subs[r]:r,
        }
        self.x_dot = sp.simplify(eq_acceleration.subs(subs))
        
        x_ = sp.Matrix(
            [u * sp.cos(psi) - v * sp.sin(psi), u * sp.sin(psi) + v * sp.cos(psi), r]
        )

        self.f_ = sp.Matrix.vstack(x_, self.x_dot)
        self.f_ = sympy.matrices.immutable.ImmutableDenseMatrix(self.f_)  # state model
        self.lambda_f = expression_to_python_method(expression=self.f_.subs(subs_simpler), function_name="lambda_f", substitute_functions=False)
        
        jac = self.f_.jacobian(X)
        h = sp.symbols("h")  # Time step
        self.Phi_ = sp.eye(len(X), len(X)) + jac * h
        self.lambda_Phi = expression_to_python_method(expression=self.Phi_.subs(subs_simpler), function_name="lambda_jacobian", substitute_functions=False)  # state transition model
        
class ExtendedKalmanFilterVMMExact(ExtendedKalmanFilterVMM):
    
    def define_prediction_model(self,model: ModularVesselSimulator, X, dynamic_symbols):
        
        eq_acceleration = model.expand_subsystemequations(model.acceleartion_eq_SI, prime=False)
        eq_acceleration = sp.simplify(eq_acceleration.subs(U,sp.sqrt(u**2+v**2)))
        
        self.lambda_acceleration = expression_to_python_method(eq_dottify(eq_acceleration.subs(subs_simpler),), function_name="acceleration", )
                
        
        ## defining the transition model:
        #x_dot = ImmutableDenseMatrix([u1d,v1d,r1d])
        
        dynamic_symbols = [u,v,r]
        dynamic_symbols_subs={symbol: me.dynamicsymbols(symbol.name) for symbol in dynamic_symbols} 

        subs={
            dynamic_symbols_subs[u].diff():u1d,
            dynamic_symbols_subs[v].diff():v1d,
            dynamic_symbols_subs[r].diff():r1d,

            dynamic_symbols_subs[u]:u,
            dynamic_symbols_subs[v]:v,
            dynamic_symbols_subs[r]:r,
        }
        self.x_dot = sp.simplify(eq_acceleration.subs(subs))
        
        #x_ = sp.Matrix(
        #    [u * sp.cos(psi) - v * sp.sin(psi), u * sp.sin(psi) + v * sp.cos(psi), r]
        #)

        x_ = sp.Matrix(
            [(u+self.x_dot[0]*h/2) * sp.cos(psi) - (v+self.x_dot[1]*h/2) * sp.sin(psi), (u+self.x_dot[0]*h/2) * sp.sin(psi) + (v+self.x_dot[1]*h/2) * sp.cos(psi), r]
        )
        
        
        self.f_ = sp.Matrix.vstack(x_, self.x_dot)
        self.f_ = sympy.matrices.immutable.ImmutableDenseMatrix(self.f_)  # state model
        self.lambda_f = expression_to_python_method(expression=self.f_.subs(subs_simpler), function_name="lambda_f", substitute_functions=False)
        
        jac = self.f_.jacobian(X)
        h = sp.symbols("h")  # Time step
        self.Phi_ = sp.eye(len(X), len(X)) + jac * h
        self.lambda_Phi = expression_to_python_method(expression=self.Phi_.subs(subs_simpler), function_name="lambda_jacobian", substitute_functions=False)  # state transition model
    