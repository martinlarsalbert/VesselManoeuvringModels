"""
References:
[1] : Wang, Tongtong, Guoyuan Li, Baiheng Wu, Vilmar Æsøy, and Houxiang Zhang. “Parameter Identification of Ship Manoeuvring Model Under Disturbance Using Support Vector Machine Method.” Ships and Offshore Structures, May 19, 2021.
"""


import sympy as sp
from vessel_manoeuvring_models.symbols import *
import pandas as pd
from vessel_manoeuvring_models.nonlinear_vmm_equations import *
from vessel_manoeuvring_models.models.vmm import Simulator
from vessel_manoeuvring_models.models.vmm import Simulator, VMM

p = df_parameters["symbol"]

subs = [
    (p.Xvdot, 0),
    (p.Xrdot, 0),
    (p.Yudot, 0),
    # (p.Yrdot,0),  # this is probably not true
    (p.Nudot, 0),
    # (p.Nvdot,0),# this is probably not true
]

## X

# Linearizing the EOM:
X_eom = X_eom.subs([(X_force, X_force), (v, 0), (r ** 2, 0)])

# [1] eq.2-a:
X_qs_eq = sp.Eq(X_D, p.Xu * u + p.Xv * v + p.Xr * r + p.Xdelta * delta)

fx_eq = fx_eq.subs(subs)
X_eq = X_eom.subs(
    [(X_force, sp.solve(fx_eq, X_force)[0]), (X_D, sp.solve(X_qs_eq, X_D)[0])]
)

## Y

# Linearizing the EOM:
Y_eom = Y_eom.subs(
    [
        (Y_force, Y_force),
        (u, U),  # Note that U is 1 in prime system!
    ]
)

# [1] eq.2-b:
Y_qs_eq = sp.Eq(Y_D, p.Yu * u + p.Yv * v + p.Yr * r + p.Ydelta * delta)

fy_eq = fy_eq.subs(subs)
Y_eq = Y_eom.subs(
    [
        (Y_force, sp.solve(fy_eq, Y_force)[0]),
        (Y_D, sp.solve(Y_qs_eq, Y_D)[0]),
    ]
)

## N

# Linearizing the EOM:
N_eom = N_eom.subs(
    [
        (N_force, N_force),
        (u, U),  # Note that U is 1 in prime system!
    ]
)

# [1] eq.2-c:
N_qs_eq = sp.Eq(N_D, p.Nu * u + p.Nv * v + p.Nr * r + p.Ndelta * delta)

mz_eq = mz_eq.subs(subs)
N_eq = N_eom.subs(
    [
        (N_force, sp.solve(mz_eq, N_force)[0]),
        (N_D, sp.solve(N_qs_eq, N_D)[0]),
    ]
)

# Create a simulator for this model:
simulator = Simulator(X_eq=X_eq, Y_eq=Y_eq, N_eq=N_eq)
simulator.define_quasi_static_forces(X_qs_eq=X_qs_eq, Y_qs_eq=Y_qs_eq, N_qs_eq=N_qs_eq)

vmm_linear = VMM(X_eq=X_eq, Y_eq=Y_eq, N_eq=N_eq)
