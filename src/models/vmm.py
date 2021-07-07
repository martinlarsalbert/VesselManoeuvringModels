"""Module for Vessel Manoeuvring Model (VMM) simulation

"""
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp
from src.symbols import *
import src.linear_vmm_equations as eq
from src import prime_system

from src.substitute_dynamic_symbols import run, lambdify
from scipy.spatial.transform import Rotation as R
from src import prime_system

eqs = [eq.X_eq, eq.Y_eq, eq.N_eq]
solution = sp.solve(eqs, u.diff(), v.diff(), r.diff(), dict=True)

## Decouple the equations:
u1d_eq = sp.Eq(u.diff(), solution[0][u.diff()]) 
v1d_eq = sp.Eq(v.diff(), solution[0][v.diff()]) 
r1d_eq = sp.Eq(r.diff(), solution[0][r.diff()]) 

## Lambdify:
subs = {value:key for key,value in eq.p.items()}
u1d_lambda = lambdify(u1d_eq.subs(subs).rhs)
v1d_lambda = lambdify(v1d_eq.subs(subs).rhs)
r1d_lambda = lambdify(r1d_eq.subs(subs).rhs)

class Simulation():

    u1d_lambda = staticmethod(u1d_lambda)
    v1d_lambda = staticmethod(v1d_lambda)
    r1d_lambda = staticmethod(r1d_lambda)
        
        
    def step(self, t, states, parameters, ship_parameters, control):

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

        inputs['U'] = np.sqrt(u**2 + v**2)  #Instantanious velocity

        u1d = run(function=self.u1d_lambda, inputs=inputs)
        v1d = run(function=self.v1d_lambda, inputs=inputs)
        r1d = run(function=self.r1d_lambda, inputs=inputs)

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

    def simulate(self, y0, t, df_parameters, ship_parameters, control, **kwargs):

        self.y0=y0
        self.t=t
        self.df_parameters = df_parameters
        self.ship_parameters = ship_parameters
        self.control = control

        ps = prime_system.PrimeSystem(**ship_parameters)
        parameters_prime = dict(df_parameters['prime'])


        # SI to prime:
        u = y0['u']
        v = y0['v']
        U = np.sqrt(u**2 + v**2)  #Initial velocity

        ship_parameters_prime = ps.prime(ship_parameters)
        control_prime = ps.prime(control)


        t_prime = ps._prime(t, unit='time', U=U)
        if isinstance(control_prime,pd.DataFrame):
            control_prime.index=t_prime

        y0_prime = ps.prime(y0, U=U)

        ## Simulate:
        t_span = [t_prime[0],t_prime[-1]]
        solution = solve_ivp(fun=self.step, t_span=t_span, y0=list(y0_prime.values()), t_eval=t_prime, 
            args=(parameters_prime, ship_parameters_prime, control_prime,), **kwargs)
        
        self.solution=solution
        return solution

    @property
    def result_prime(self):
        assert hasattr(self,'solution')
        
        columns = list(self.y0.keys())
        df_result_prime = pd.DataFrame(data=self.solution.y.T, columns=columns)
        df_result_prime.index=self.t[0:len(df_result_prime)]
        return df_result_prime

    @property
    def result(self):
        
        ps = prime_system.PrimeSystem(**self.ship_parameters)  # model

        U_ = np.sqrt(self.y0['u']**2 + self.y0['v']**2)
        df_result = ps.unprime(values=self.result_prime, U=U_)
        df_result['beta'] = -np.arctan2(df_result['v'],df_result['u'])


