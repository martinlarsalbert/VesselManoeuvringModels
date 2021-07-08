"""Module for Linear Vessel Manoeuvring Model (LVMM)
"""
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp
from src.symbols import *
from src.parameters import *
import src.linear_vmm_equations as eq
from src import prime_system

from src.substitute_dynamic_symbols import run, lambdify
from scipy.spatial.transform import Rotation as R
from src.models.vmm import Simulation

eqs = [eq.X_eq, eq.Y_eq, eq.N_eq]
solution = sp.solve(eqs, u.diff(), v.diff(), r.diff(), dict=True)

### Decouple the equations:
#u1d_eq = sp.Eq(u.diff(), solution[0][u.diff()]) 
#v1d_eq = sp.Eq(v.diff(), solution[0][v.diff()]) 
#r1d_eq = sp.Eq(r.diff(), solution[0][r.diff()]) 
#
### Lambdify:
#subs = {value:key for key,value in eq.p.items()}
#u1d_lambda = lambdify(u1d_eq.subs(subs).rhs)
#v1d_lambda = lambdify(v1d_eq.subs(subs).rhs)
#r1d_lambda = lambdify(r1d_eq.subs(subs).rhs)

subs=[
    (x_G,0),
    (eq.p.Xvdot,0),
    (eq.p.Xrdot,0),
    (eq.p.Yudot,0),
    (eq.p.Yrdot,0),
    (eq.p.Nudot,0),
    (eq.p.Nvdot,0),   
]
u1d_eq = sp.Eq(u.diff(),sp.solve(eq.X_eq,u.diff())[0])
v1d_eq = sp.Eq(v.diff(),sp.solve(eq.Y_eq,v.diff())[0])
r1d_eq = sp.Eq(r.diff(),sp.solve(eq.N_eq,r.diff())[0])
u1d_eq=u1d_eq.subs(subs)
v1d_eq=v1d_eq.subs(subs)
r1d_eq=r1d_eq.subs(subs)

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
    
    if isinstance(control, pd.DataFrame):
        index = np.argmin(np.array(np.abs(control.index - t)))
        control_ = dict(control.iloc[index])
    else:
        control_ = control
    
    inputs.update(control_)

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

def simulate(y0, t, df_parameters, ship_parameters, control, **kwargs):
        
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
    solution = solve_ivp(fun=step, t_span=t_span, y0=list(y0_prime.values()), t_eval=t_prime, args=(parameters_prime, ship_parameters_prime, control_prime,), **kwargs)
    #assert solution.success
    
    return solution


class LinearSimulation(Simulation):

    u1d_lambda = staticmethod(u1d_lambda)
    v1d_lambda = staticmethod(v1d_lambda)
    r1d_lambda = staticmethod(r1d_lambda)