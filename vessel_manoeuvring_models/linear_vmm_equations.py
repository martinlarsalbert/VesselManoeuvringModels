"""
References:
[1] : Matusiak, Jerzy. Dynamics of a Rigid Ship - with Applications, 3rd Edition, 2021.
[2] : Triantafyllou, Michael S, and Franz S Hover. “MANEUVERING AND CONTROL OF MARINE VEHICLES.” Massachusetts Institute of Technology, 2003, 152.
"""


import sympy as sp
from vessel_manoeuvring_models.symbols import *
from vessel_manoeuvring_models.parameters import *
import vessel_manoeuvring_models.nonlinear_vmm_equations as nonlinear_vmm_equations
import pandas as pd

p = df_parameters["symbol"]

## X

# Linearizing the EOM:
X_eom = nonlinear_vmm_equations.X_eom.subs([(X_force, X_force), (v, 0), (r ** 2, 0)])

X_qs_eq = sp.Eq(X_D, p.Xu * u + p.Xv * v + +p.Xr * r + p.Xdelta * delta)

fx_eq = sp.Eq(X_force, p.Xudot * u1d + p.Xvdot * v1d + p.Xrdot * r1d + X_D)

X_eq = X_eom.subs(
    [
        (X_force, sp.solve(fx_eq, X_force)[0]),
        (X_D, sp.solve(X_qs_eq, X_D)[0]),
    ]
)

## Y

# Linearizing the EOM:
Y_eom = nonlinear_vmm_equations.Y_eom.subs(
    [
        (Y_force, Y_force),
        (u, U),  # Note that U is 1 in prime system!
    ]
)

Y_qs_eq = sp.Eq(Y_D, p.Yu * u + p.Yv * v + p.Yr * r + p.Ydelta * delta)

fy_eq = sp.Eq(Y_force, p.Yudot * u1d + p.Yvdot * v1d + p.Yrdot * r1d + Y_D)

Y_eq = Y_eom.subs(
    [
        (Y_force, sp.solve(fy_eq, Y_force)[0]),
        (Y_D, sp.solve(Y_qs_eq, Y_D)[0]),
    ]
)

## N

# Linearizing the EOM:
N_eom = nonlinear_vmm_equations.N_eom.subs(
    [
        (N_force, N_force),
        (u, U),  # Note that U is 1 in prime system!
    ]
)

N_qs_eq = sp.Eq(N_D, p.Nu * u + p.Nv * v + p.Nr * r + p.Ndelta * delta)

mz_eq = sp.Eq(N_force, p.Nudot * u1d + p.Nvdot * v1d + p.Nrdot * r1d + N_D)

N_eq = N_eom.subs(
    [(N_force, sp.solve(mz_eq, N_force)[0]), (N_D, sp.solve(N_qs_eq, N_D)[0])]
)
