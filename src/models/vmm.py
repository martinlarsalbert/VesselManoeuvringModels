"""Module for Vessel Manoeuvring Model (VMM) simulation

References:
[1] Matusiak, Jerzy. Dynamics of a Rigid Ship - with Applications, 3rd Edition, 2021.


"""

import pandas as pd
import numpy as np

from scipy.integrate import solve_ivp
import sympy as sp
from src.symbols import *
import warnings

from src.parameters import df_parameters
from src.prime_system import PrimeSystem
p = df_parameters['symbol']


from src.substitute_dynamic_symbols import run, lambdify
from scipy.spatial.transform import Rotation as R
import src.prime_system

from src.models.regression import get_coefficients
import dill
from src.models.result import Result

class Simulator():


    def __init__(self,X_eq,Y_eq,N_eq):

        self.X_eq = X_eq
        self.Y_eq = Y_eq
        self.N_eq = N_eq

    def define_EOM(self,X_eq:sp.Eq, Y_eq:sp.Eq, N_eq:sp.Eq):
        """Define equation of motion

        Args:
            X_eq (sp.Eq): [description]
            Y_eq (sp.Eq): [description]
            N_eq (sp.Eq): [description]
        """

    
        u1d,v1d,r1d = sp.symbols('u1d, v1d, r1d')
    
        subs = [
            (u.diff(),u1d),
            (v.diff(),v1d),
            (r.diff(),r1d),

        ]
        eq_X_ = X_eq.subs(subs)
        eq_Y_ = Y_eq.subs(subs)
        eq_N_ = N_eq.subs(subs)

        A,b = sp.linear_eq_to_matrix([eq_X_,eq_Y_,eq_N_],[u1d,v1d,r1d])
        self.A = A
        self.b = b
        self.acceleartion_eq = A.inv()*b
        
        ## Lambdify:
        subs = {value:key for key,value in p.items()}
        subs[X_qs] = sp.symbols('X_qs')
        subs[Y_qs] = sp.symbols('Y_qs')
        subs[N_qs] = sp.symbols('N_qs')

        acceleration_lambda = lambdify(self.acceleartion_eq.subs(subs))  
        return acceleration_lambda      

    def define_quasi_static_forces(self, X_qs_eq:sp.Eq, Y_qs_eq:sp.Eq, N_qs_eq:sp.Eq):
        """Define the equations for the quasi static forces
        Ex:
        Y_qs(u,v,r,delta) = Yuu*u**2 + Yv*v + Yr*r + p.Ydelta*delta + ...

        Args:
            X_qs_eq (sp.Eq): [description]
            Y_qs_eq (sp.Eq): [description]
            N_qs_eq (sp.Eq): [description]
        """
        self.X_qs_eq = X_qs_eq
        self.Y_qs_eq = Y_qs_eq
        self.N_qs_eq = N_qs_eq
        
    def get_all_coefficients(self, sympy_symbols=True):
        return (
            self.get_coefficients_X(sympy_symbols=sympy_symbols)
            + self.get_coefficients_Y(sympy_symbols=sympy_symbols)
            + self.get_coefficients_N(sympy_symbols=sympy_symbols)
                    
        )
    
    def get_coefficients_X(self,sympy_symbols=True):
        eq = self.X_eq.subs(X_qs, self.X_qs_eq.rhs)
        return self._get_coefficients(eq=eq, sympy_symbols=sympy_symbols)
        

    def get_coefficients_Y(self,sympy_symbols=True):
        eq = self.Y_eq.subs(Y_qs, self.Y_qs_eq.rhs)
        return self._get_coefficients(eq=eq, sympy_symbols=sympy_symbols)

    def get_coefficients_N(self,sympy_symbols=True):
        eq = self.N_eq.subs(N_qs, self.N_qs_eq.rhs)
        return self._get_coefficients(eq=eq, sympy_symbols=sympy_symbols)
                
    @staticmethod
    def _get_coefficients(eq,sympy_symbols=True):
        
        coefficients = get_coefficients(eq=eq, base_features=[u,v,r,delta,thrust])

        if sympy_symbols:
            return coefficients
        else:
            subs = {value:key for key,value in p.items()}
            string_coefficients = [subs[coefficient]  for coefficient in coefficients]
            return string_coefficients

    def step(self, t:float, states:np.ndarray, parameters:dict, ship_parameters:dict, control:pd.DataFrame, U0=1)->np.ndarray:
        """ Calculate states derivatives for next time step


        Parameters
        ----------
        t : float
            current time
        states : np.ndarray
            current states as a vector
        parameters : dict
            hydrodynamic derivatives
        ship_parameters : dict
            ship parameters lpp, beam, etc.
        control : pd.DataFrame
            data frame with time series for control devices such as rudder angle (delta) and popeller thrust.
        U0 : float
            initial velocity constant [1] (only used for linearized models)

        Returns
        -------
        np.ndarray
            states derivatives for next time step
        """

        u,v,r,x0,y0,psi = states

        states_dict = {
            'u':u,
            'v':v,
            'r':r,
            'x0':x0,
            'y0':y0,
            'psi':psi,

            }

        inputs = dict(parameters)
        inputs.update(ship_parameters)
        inputs.update(states_dict)

        if isinstance(control, pd.DataFrame):
            index = np.argmin(np.array(np.abs(control.index - t)))
            control_ = dict(control.iloc[index])
        else:
            control_ = control
        inputs.update(control_)

        inputs['U'] = U0  # initial velocity constant [1]

        inputs['X_qs'] = run(function=self.X_qs_lambda, inputs=inputs)
        inputs['Y_qs'] = run(function=self.Y_qs_lambda, inputs=inputs)
        inputs['N_qs'] = run(function=self.N_qs_lambda, inputs=inputs)
        u1d,v1d,r1d = run(function=self.acceleration_lambda, inputs=inputs)
        
        # get rid of brackets:
        u1d=u1d[0]
        v1d=v1d[0]
        r1d=r1d[0]

        rotation = R.from_euler('z', psi, degrees=False)
        w = 0
        velocities = rotation.apply([u,v,w])
        x01d = velocities[0]
        y01d = velocities[1]
        psi1d = r    
        dstates = [
            u1d,
            v1d,
            r1d,
            x01d,
            y01d,
            psi1d,
        ]    
        return dstates

    def step_primed_parameters(self, t, states, parameters, ship_parameters, control, U0):
        """
        The simulation is carried out with states in SI units.
        The parameters are often expressed in prime system.
        This means that:
        1) the previous state needs to be converted to prime
        2) dstate is calculate in prime system
        3) dstate is converted back to SI.

        Args:
            t ([type]): [description]
            states ([type]): [description]
            parameters ([type]): [description]
            ship_parameters ([type]): [description]
            control ([type]): [description]
        """

        # 1)
        u,v,r,x0,y0,psi = states
        U = np.sqrt(u**2 + v**2)  #Instantanious velocity
        states_dict = {
            'u':u,
            'v':v,
            'r':r,
            'x0':x0,
            'y0':y0,
            'psi':psi,
        }
        states_dict_prime = self.prime_system.prime(states_dict, U=U)

        df_control_prime = self.prime_system.prime(control, U=U)

        # 2)
        dstates_prime = self.step(t=t, 
            states=list(states_dict_prime.values()), 
            parameters=parameters, 
            ship_parameters=self.ship_parameters_prime, 
            control=df_control_prime, U0=1) # Note that U0 is 1 in prime system!

        # 3)
        u1d_prime,v1d_prime,r1d_prime,x01d_prime,y01d_prime,psi1d_prime = dstates_prime
        
        states_dict_prime = {
            'u1d':u1d_prime,
            'v1d':v1d_prime,
            'r1d':r1d_prime,
            'x01d':x01d_prime,
            'y01d':y01d_prime,
            'psi1d':psi1d_prime,
        }

        dstates_dict = self.prime_system.unprime(states_dict_prime, U=U)
        dstates = list(dstates_dict.values())
        return dstates

    def simulate(self, df_, parameters, ship_parameters, control_keys=['delta','thrust'], 
        primed_parameters=False, prime_system=None, method='Radau',
        name='simulation', additional_events=[], **kwargs):

        self.acceleration_lambda = self.define_EOM(X_eq=self.X_eq, Y_eq=self.Y_eq, N_eq=self.N_eq)
        subs = {value:key for key,value in p.items()}        
        self.X_qs_lambda = lambdify(self.X_qs_eq.rhs.subs(subs))
        self.Y_qs_lambda = lambdify(self.Y_qs_eq.rhs.subs(subs))
        self.N_qs_lambda = lambdify(self.N_qs_eq.rhs.subs(subs))

        t = df_.index
        t_span = [t.min(),t.max()]
        t_eval = np.linspace(t.min(),t.max(),len(t))

        df_control = df_[control_keys]
        
        self.primed_parameters = primed_parameters
        if primed_parameters:
            self.prime_system = prime_system
            assert isinstance(self.prime_system, src.prime_system.PrimeSystem)
            self.ship_parameters_prime = self.prime_system.prime(ship_parameters)
            step = self.step_primed_parameters
        else:
            step = self.step
        

        df_0 = df_.iloc[0:5].mean(axis=0)
        y0 = {
            'u' : df_0['u'], 
            'v' : df_0['v'],
            'r' : df_0['r'],
            'x0' : df_0['x0'],
            'y0' : df_0['y0'],
            'psi' : df_0['psi']
            }
        U0 = np.sqrt(df_0['u']**2 + df_0['v']**2)  # initial velocity constant [1]

        def stoped(t, states, parameters, ship_parameters, control, U0): 
            u,v,r,x0,y0,psi = states
            return u
        stoped.terminal = True
        stoped.direction = -1

        def drifting(t, states, parameters, ship_parameters, control, U0): 
            u,v,r,x0,y0,psi = states
            
            beta = np.deg2rad(70) - np.abs(-np.arctan2(v, u))
            
            return beta
        drifting.terminal = True
        drifting.direction = -1
        events = [stoped, drifting] + additional_events

        solution = solve_ivp(fun=step, t_span=t_span, y0=list(y0.values()), t_eval=t_eval, 
                    args=(parameters, ship_parameters, df_control, U0), method=method, events=events, **kwargs)
        
        if not solution.success:
            #warnings.warn(solution.message)
            raise ValueError(solution.message)

        
        result = Result(simulator=self, solution=solution, df_model_test=df_, df_control=df_control, 
                    ship_parameters=ship_parameters, parameters=parameters, y0=y0, name=name)
        return result


class ModelSimulator(Simulator):
    """Ship and parameter specific simulator.
    """

    def __init__(self,simulator:Simulator, parameters:dict, ship_parameters:dict, control_keys:list, 
            prime_system:PrimeSystem, name='simulation', primed_parameters=True, method='Radau'):
        """Generate a simulator that is specific to one ship with a specific set of parameters.
        This is done by making a copy of an existing simulator object and add freezed parameters.

        Parameters
        ----------
        simulator : Simulator
            Simulator object with predefined odes
        parameters : dict
            [description]
        ship_parameters : dict
            [description]
        control_keys : list
            [description]
        prime_system : PrimeSystem
            [description]
        name : str, optional
            [description], by default 'simulation'
        primed_parameters : bool, optional
            [description], by default True
        method : str, optional
            [description], by default 'Radau'
        """
        
        self.__dict__.update(simulator.__dict__)
        self.parameters = self.extract_needed_parameters(parameters)
        self.ship_parameters = ship_parameters
        self.control_keys = control_keys
        self.primed_parameters = primed_parameters
        self.prime_system = prime_system
        self.name = name

    def extract_needed_parameters(self, parameters:dict)->dict:
        
        coefficients=self.get_all_coefficients(sympy_symbols=False)
        parameters = pd.Series(parameters).dropna()
        
        missing_coefficients = set(coefficients) - set(parameters.keys())
        assert len(missing_coefficients)==0, f'Missing parameters:{missing_coefficients}'

        return parameters[coefficients]

    def simulate(self, df_, method='Radau', name='simulaton', additional_events=[], **kwargs)->Result:
        
        return super().simulate(df_=df_, parameters=self.parameters, ship_parameters=self.ship_parameters, control_keys=self.control_keys, 
                primed_parameters=self.primed_parameters, prime_system=self.prime_system, method=method, name=name, 
                additional_events=additional_events, **kwargs)

    def turning_circle(self, u0:float, angle:float=35.0, t_max:float=1000.0, dt:float=0.1, method='Radau', name='simulation', **kwargs)->Result:
        """Turning circle simulation

        Parameters
        ----------
        u0 : float
            initial speed [m/s]
        angle : float, optional
            Rudder angle [deg], by default 35.0 [deg]
        t_max : float, optional
            max simulation time, by default 1000.0
        dt : float, optional
            time step, by default 0.1
        method : str, optional
            Method to solve ivp see solve_ivp, by default 'Radau'
        name : str, optional
            [description], by default 'simulation'

        Returns
        -------
        Result
            [description]
        """

        t_ = np.arange(0,t_max, dt)
        df_ = pd.DataFrame(index=t_)
        df_['x0'] = 0
        df_['y0'] = 0
        df_['psi'] = 0
        df_['u'] = u0
        df_['v'] = 0
        df_['r'] = 0
        df_['delta'] = np.deg2rad(angle)

        def completed(t, states, parameters, ship_parameters, control, U0):
            u,v,r,x0,y0,psi = states
            remain = np.deg2rad(360) - np.abs(psi)
            return remain
        completed.terminal = True
        completed.direction = -1

        additional_events = [
            completed,
        ]

        return self.simulate(df_=df_, method=method, name=name, additional_events=additional_events, **kwargs)

    def save(self, path:str):
        """Save model to pickle file

        Parameters
        ----------
        path : str
            Ex:'model.pkl'
        """

        with open(path, mode='wb') as file:
            dill.dump(self, file=file)

    def __getstate__(self):
        def should_pickle(k):
            return not k in ['acceleration_lambda','X_qs_lambda','Y_qs_lambda','N_qs_lambda']
                
        return {k:v for (k, v) in self.__dict__.items() if should_pickle(k)}

    @classmethod
    def load(cls, path:str):

        with open(path, mode='rb') as file:
            obj = dill.load(file=file)
        
        return obj






        


