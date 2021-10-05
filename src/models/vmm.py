"""Module for Vessel Manoeuvring Model (VMM) simulation

References:
[1] Matusiak, Jerzy. Dynamics of a Rigid Ship - with Applications, 3rd Edition, 2021.


"""
from os import stat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
from src.visualization.plot import track_plot
from src.models.regression import get_coefficients
from copy import deepcopy

class Simulator():


    def __init__(self,X_eq,Y_eq,N_eq):

        self.X_eq = X_eq
        self.Y_eq = Y_eq
        self.N_eq = N_eq

        self.define_EOM(X_eq=X_eq, Y_eq=Y_eq, N_eq=N_eq)


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

        self.acceleration_lambda = lambdify(self.acceleartion_eq.subs(subs))        

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
        
        subs = {value:key for key,value in p.items()}        
        self.X_qs_lambda = lambdify(X_qs_eq.rhs.subs(subs))
        self.Y_qs_lambda = lambdify(Y_qs_eq.rhs.subs(subs))
        self.N_qs_lambda = lambdify(N_qs_eq.rhs.subs(subs))
        
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
        name='simulation',**kwargs):

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

        solution = solve_ivp(fun=step, t_span=t_span, y0=list(y0.values()), t_eval=t_eval, 
                    args=(parameters, ship_parameters, df_control, U0), method=method, **kwargs)
        
        if not solution.success:
            #warnings.warn(solution.message)
            raise ValueError(solution.message)

        
        result = Result(simulator=self, solution=solution, df_model_test=df_, df_control=df_control, 
                    ship_parameters=ship_parameters, parameters=parameters, y0=y0, name=name)
        return result

class ModelSimulator(Simulator):

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

    def simulate(self, df_, method='Radau', name='simulaton', **kwargs):
        
        return super().simulate(df_=df_, parameters=self.parameters, ship_parameters=self.ship_parameters, control_keys=self.control_keys, 
                primed_parameters=self.primed_parameters, prime_system=self.prime_system, method=method, name=name, **kwargs)


class Result():

    def __init__(self, simulator, solution, df_model_test, df_control, ship_parameters, parameters, y0, 
    include_accelerations=True, name='simulation'):
    
        self.simulator = simulator
        self.solution=solution
        self.df_model_test=df_model_test
        self.df_control = df_control
        self.ship_parameters = ship_parameters
        self.parameters = parameters
        self.y0=y0
        self.include_accelerations = include_accelerations
        self.name=name

    @property
    def simulation_result(self):

        columns = list(self.y0.keys())
        df_result = pd.DataFrame(data=self.solution.y.T, columns=columns, index=self.solution.t)
        df_result = pd.merge(left=df_result, right=self.df_control, how='left', 
            left_index=True, right_index=True) 

        try:
            df_result['beta'] = -np.arctan2(df_result['v'],df_result['u'])
        except:
            pass

        try:
            df_result['U'] = np.sqrt(df_result['u']**2 + df_result['v']**2)
        except:
            pass

        return df_result

    @property
    def result(self):
        df_result = self.simulation_result
        
        if self.include_accelerations:
            df_result = pd.concat([df_result,self.accelerations], axis=1)

        return df_result

    @property
    def X_qs(self)->pd.Series:
        """Hydrodynamic force from ship in X-direction during simulation"""
        return self._calcualte_qs_force(function=self.simulator.X_qs_lambda, unit='force')

    @property
    def Y_qs(self)->pd.Series:
        """Hydrodynamic force from ship in Y-direction during simulation"""
        return self._calcualte_qs_force(function=self.simulator.Y_qs_lambda, unit='force')

    @property
    def N_qs(self)->pd.Series:
        """Hydrodynamic force from ship in N-direction during simulation"""
        return self._calcualte_qs_force(function=self.simulator.N_qs_lambda, unit='moment')

    def _calcualte_qs_force(self, function, unit):
        df_result = self.simulation_result.copy()
        
        if self.simulator.primed_parameters:
            df_result_prime = self.simulator.prime_system.prime(df_result, U=df_result['U'])
            X_qs_ = run(function=function, inputs=df_result_prime, **self.parameters)
            return self.simulator.prime_system._unprime(X_qs_, unit=unit, U=df_result['U'])
        else:
            return run(function=function, inputs=df_result, **self.parameters)

    @property
    def accelerations(self):
        df_result = self.simulation_result.copy()
        
        if self.simulator.primed_parameters:
            df_result_prime = self.simulator.prime_system.prime(df_result, 
                                                                U=df_result['U'])

            inputs = df_result_prime
            inputs['U0'] = inputs.iloc[0]['U']

            u1d_prime,v1d_prime,r1d_prime = run(function=self.simulator.acceleration_lambda, 
                X_qs=run(function=self.simulator.X_qs_lambda, inputs=inputs, **self.parameters),
                Y_qs=run(function=self.simulator.Y_qs_lambda, inputs=inputs, **self.parameters),
                N_qs=run(function=self.simulator.N_qs_lambda, inputs=inputs, **self.parameters),
                inputs=inputs, 
                **self.parameters,
                **self.simulator.ship_parameters_prime)

            df_accelerations_prime = pd.DataFrame(index=df_result.index)
            df_accelerations_prime['u1d'] = u1d_prime[0]
            df_accelerations_prime['v1d'] = v1d_prime[0]
            df_accelerations_prime['r1d'] = r1d_prime[0]
            df_accelerations = self.simulator.prime_system.unprime(df_accelerations_prime, U=df_result['U'])
        else:
            
            inputs = df_result
            inputs['U0'] = inputs.iloc[0]['U']
            
            u1d,v1d,r1d = run(function=self.simulator.acceleration_lambda, 
                X_qs=run(function=self.simulator.X_qs_lambda, inputs=inputs, **self.parameters),
                Y_qs=run(function=self.simulator.Y_qs_lambda, inputs=inputs, **self.parameters),
                N_qs=run(function=self.simulator.N_qs_lambda, inputs=inputs, **self.parameters),
                inputs=inputs,
                **self.parameters, 
                **self.ship_parameters)

            df_accelerations = pd.DataFrame(index=df_result.index)
            df_accelerations['u1d'] = u1d[0]
            df_accelerations['v1d'] = v1d[0]
            df_accelerations['r1d'] = r1d[0]       

        return df_accelerations
    
    def plot_compare(self, compare=True):

        self.track_plot(compare=compare)
        self.plot(compare=compare)

    def track_plot(self,ax=None, compare=True):
        if ax is None:
            fig,ax = plt.subplots()

        track_plot(df=self.simulation_result, lpp=self.ship_parameters['L'], beam=self.ship_parameters['B'],ax=ax, 
            label=self.name, color='green')
        
        if compare:
            track_plot(df=self.df_model_test, lpp=self.ship_parameters['L'], beam=self.ship_parameters['B'],ax=ax, label='data')
            ax.legend()
        return ax

    def plot(self, ax=None, subplot=True, compare=True):
        
        df_result = self.simulation_result

        if subplot:
            number_of_axes = len(df_result.columns)
            ncols=2
            nrows = int(np.ceil(number_of_axes / ncols))
            fig,axes=plt.subplots(ncols=ncols, nrows=nrows)
            axes = axes.flatten()

        for i,key in enumerate(df_result):
            if subplot:
                ax = axes[i]
            else:
                fig,ax = plt.subplots()
            
            df_result.plot(y=key, label=self.name, ax=ax)
            
            if compare:
                self.df_model_test.plot(y=key, label='data', style='--', ax=ax)

            ax.get_legend().set_visible(False)
            ax.set_ylabel(key)
        
        axes[0].legend()

        plt.tight_layout()
        return fig

    def save(self,path:str):
        """Save the simulation to a csv file"""
        self.result.to_csv(path, index=True)
    


        


