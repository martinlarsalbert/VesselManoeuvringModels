"""Module for Linear Vessel Manoeuvring Model (LVMM)
"""
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp
from src.symbols import *
import src.linear_vmm_equations as eq

from src.substitute_dynamic_symbols import run, lambdify
from scipy.spatial.transform import Rotation as R

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
    inputs.update(control)
    inputs['U'] = np.sqrt(u**2 + v**2)  #Instantanious velocity
    
    u1d = run(function=u1d_lambda, inputs=inputs)
    v1d = run(function=v1d_lambda, inputs=inputs)
    r1d = run(function=r1d_lambda, inputs=inputs)
    
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

def simulate(y0, t, df_parameters, df_ship_parameters, control, **kwargs):
    
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
        df_prime.length,
        df_prime.length,
        df_prime.angle,
    ]
    y0_prime = [y/run(prime['lambda'], inputs_prime) for prime, y in zip(primes, y0)]
    
    control_prime = {key:value[0]/run(function=df_prime.loc['lambda',value[1]], inputs=inputs_prime) for key,value in control.items()}
    
    t_span = [t[0],t[-1]]
    
    solution = solve_ivp(fun=step, t_span=t_span, y0=y0_prime, t_eval=t_prime, args=(parameters, ship_parameters, control_prime,), **kwargs)
    #assert solution.success
    
    return solution


