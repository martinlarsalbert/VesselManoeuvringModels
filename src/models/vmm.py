"""Module for Vessel Manoeuvring Model (VMM) simulation

References:
[1] Matusiak, Jerzy. Dynamics of a Rigid Ship - with Applications, 3rd Edition, 2021.


"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sp
from src.symbols import *

from src.parameters import df_parameters
p = df_parameters['symbol']


from src.substitute_dynamic_symbols import run, lambdify
from scipy.spatial.transform import Rotation as R
import src.prime_system
from src.visualization.plot import track_plot

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

        # 2)
        dstates_prime = self.step(t=t, states=list(states_dict_prime.values()), parameters=parameters, ship_parameters=self.ship_parameters_prime, 
            control=self.df_control_prime, U0=1) # Note that U0 is 1 in prime system!

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


    def simulate(self, df_, parameters, ship_parameters, control_keys=['delta','thrust'], primed_parameters=False, prime_system=None):

        t = df_.index
        t_span = [t.min(),t.max()]
        t_eval = np.linspace(t.min(),t.max(),len(t))

        control_keys = control_keys
        df_control = df_[control_keys]
        
        if primed_parameters:
            self.primed_parameters = primed_parameters
            self.prime_system = prime_system
            assert isinstance(self.prime_system, src.prime_system.PrimeSystem)
            self.ship_parameters_prime = self.prime_system.prime(ship_parameters)
            self.df_control_prime = self.prime_system.prime(df_control, U=df_['U'])
            step = self.step_primed_parameters
        else:
            step = self.step
        
        ship_parameters = ship_parameters
        parameters = parameters

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
                    args=(parameters, ship_parameters, df_control, U0))
        
        
        result = Result(simulator=self, solution=solution, df_model_test=df_, df_control=df_control, ship_parameters=ship_parameters, parameters=parameters, y0=y0)
        return result
    
class Result():

    def __init__(self, simulator, solution, df_model_test, df_control, ship_parameters, parameters, y0):
        self.simulator = simulator
        self.solution=solution
        self.df_model_test=df_model_test
        self.df_control = df_control
        self.ship_parameters = ship_parameters
        self.parameters = parameters
        self.y0=y0

    @property
    def result(self):

        columns = list(self.y0.keys())
        df_result = pd.DataFrame(data=self.solution.y.T, columns=columns, index=self.solution.t)
        df_result[self.df_control.columns] = self.df_control.values

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
        df_result = self.result.copy()
        
        if self.simulator.primed_parameters:
            df_result_prime = self.simulator.prime_system.prime(df_result, U=df_result['U'])
            X_qs_ = run(function=function, inputs=df_result_prime, **self.parameters)
            return self.simulator.prime_system._unprime(X_qs_, unit=unit, U=df_result['U'])
        else:
            return run(function=function, inputs=df_result, **self.parameters)

    @property
    def accelerations(self):
        df_result = self.result.copy()
        
        if self.simulator.primed_parameters:
            df_result_prime = self.simulator.prime_system.prime(df_result, U=df_result['U'])

            inputs = df_result_prime
            inputs['U'] = inputs.iloc[0]['U']

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
            inputs['U'] = inputs.iloc[0]['U']
            
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
        

    def plot_compare(self):

        df_result = self.result
        fig,ax = plt.subplots()
        track_plot(df=self.df_model_test, lpp=self.ship_parameters['L'], beam=self.ship_parameters['B'],ax=ax, label='model test')
        track_plot(df=df_result, lpp=self.ship_parameters['L'], beam=self.ship_parameters['B'],ax=ax, label='simulation', color='green', linestyle='--')
        ax.legend()

        for key in df_result:
            fig,ax = plt.subplots()
            self.df_model_test.plot(y=key, label='model test', ax=ax)
            df_result.plot(y=key, label='simulation', style='--', ax=ax)
            ax.set_ylabel(key)

    def track_plot(self,ax=None):
        if ax is None:
            fig,ax = plt.subplots()

        track_plot(df=self.result, lpp=self.ship_parameters['L'], beam=self.ship_parameters['B'],ax=ax, label='simulation', color='green')

        return ax

    def plot(self, ax=None):
        
        df_result = self.result

        for key in df_result:
            fig,ax = plt.subplots()
            df_result.plot(y=key, label='simulation', ax=ax)
            ax.set_ylabel(key)
    


        


