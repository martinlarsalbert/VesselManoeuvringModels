"""Module for Linear Vessel Manoeuvring Model (LVMM)
"""
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp
from vessel_manoeuvring_models.symbols import *
import vessel_manoeuvring_models.nonlinear_vmm_equations as eq
from vessel_manoeuvring_models import prime_system

from vessel_manoeuvring_models.substitute_dynamic_symbols import run, lambdify
from scipy.spatial.transform import Rotation as R
from vessel_manoeuvring_models.models.vmm import Simulator

eqs = [eq.X_eq, eq.Y_eq, eq.N_eq]
solution = sp.solve(eqs, u1d, v1d, r1d, dict=True)

## Decouple the equations:
u1d_eq = sp.Eq(u1d, solution[0][u1d])
v1d_eq = sp.Eq(v1d, solution[0][v1d])
r1d_eq = sp.Eq(r1d, solution[0][r1d])

## Lambdify:
subs = {value: key for key, value in eq.p.items()}
u1d_lambda = lambdify(u1d_eq.subs(subs).rhs)
v1d_lambda = lambdify(v1d_eq.subs(subs).rhs)
r1d_lambda = lambdify(r1d_eq.subs(subs).rhs)


class NonLinearSimulation(Simulator):

    u1d_lambda = staticmethod(u1d_lambda)
    v1d_lambda = staticmethod(v1d_lambda)
    r1d_lambda = staticmethod(r1d_lambda)
