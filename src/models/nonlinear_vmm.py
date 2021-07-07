"""Module for Linear Vessel Manoeuvring Model (LVMM)
"""
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp
from src.symbols import *
import src.nonlinear_vmm_equations as eq
from src import prime_system

from src.substitute_dynamic_symbols import run, lambdify
from scipy.spatial.transform import Rotation as R
from src.models.vmm import Simulation

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


class NonLinearSimulation(Simulation):

    u1d_lambda = staticmethod(u1d_lambda)
    v1d_lambda = staticmethod(v1d_lambda)
    r1d_lambda = staticmethod(r1d_lambda)