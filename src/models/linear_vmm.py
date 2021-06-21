"""Module for Linear Vessel Manoeuvring Model (LVMM)
"""
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp
from src.symbols import *
import src.linear_vmm_equations as eq

from src.substitute_dynamic_symbols import run, lambdify


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

def step(t, states, parameters, ship_parameters, control):
    
    states_dict = {
        'u':states[0],
        'v':states[1],
        'r':states[2],
        #'u1d':states[3],
        #'v1d':states[4],
        #'r1d':states[5],
                   }
    
    inputs = dict(parameters)
    inputs.update(ship_parameters)
    inputs.update(states_dict)
    inputs.update(control)
    inputs['U'] = np.sqrt(states[0]**2 + states[1]**2)  #Instantanious velocity
    
    u1d = run(function=u1d_lambda, inputs=inputs)
    v1d = run(function=v1d_lambda, inputs=inputs)
    r1d = run(function=r1d_lambda, inputs=inputs)
    
    dstates = [
        u1d,
        v1d,
        r1d,
        #1,
        #1,
        #1,
    ]    
    
    return dstates

def simulate(y0, t, df_parameters, df_ship_parameters, control):
    
    parameters = dict(df_parameters['prime'])
    ship_parameters = dict(df_ship_parameters.loc['prime'])
    
    # SI to prime:
    inputs_prime = dict(df_ship_parameters.loc['value'])
    inputs_prime['U'] = np.sqrt(y0[0]**2 + y0[1]**2)  #Initial velocity
    
    t_prime = t.copy()
    t_prime/=run(function=df_prime.time['lambda'], inputs=inputs_prime)
    primes = [
        df_prime.linear_velocity,
        df_prime.linear_velocity,
        df_prime.angular_velocity,
        #df_prime.linear_acceleration,
        #df_prime.linear_acceleration,
        #df_prime.angular_acceleration,
    ]
    y0_prime = [y/run(prime['lambda'], inputs_prime) for prime, y in zip(primes, y0)]
    
    control_prime = {key:value[0]/run(function=df_prime.loc['lambda',value[1]], inputs=inputs_prime) for key,value in control.items()}
    
    t_span = [t[0],t[-1]]
    
    solution = solve_ivp(fun=step, t_span=t_span, y0=y0_prime, t_eval=t_prime, args=(parameters, ship_parameters, control_prime,) )
    assert solution.success
    
    return solution


