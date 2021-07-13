"""
References:
[1] : Matusiak, Jerzy. Dynamics of a Rigid Ship - with Applications, 3rd Edition, 2021.
[2] : Triantafyllou, Michael S, and Franz S Hover. “MANEUVERING AND CONTROL OF MARINE VEHICLES.” Massachusetts Institute of Technology, 2003, 152.
"""


import sympy as sp
from src.symbols import *
from src.parameters import
import src.nonlinear_vmm_equations as nonlinear_vmm_equations
import pandas as pd

p = df_parameters['symbol']

## X

# Linearizing the EOM:
X_eom = nonlinear_vmm_equations.X_eom.subs(
    [
        (X_nonlin,X_lin),
        (v,0),
        (r**2,0)
    ]
)

X_qs_eq = sp.Eq(X_qs,
        p.Xu*u + p.Xv*v + + p.Xr*r + p.Xdelta*delta)

fx_eq = sp.Eq(X_lin,
             p.Xudot*u.diff() +  p.Xvdot*v.diff() +  p.Xrdot*r.diff() + X_qs)

X_eq = X_eom.subs([
    (X_lin,sp.solve(fx_eq,X_lin)[0]),
    (X_qs,sp.solve(X_qs_eq,X_qs)[0]),
    ])

## Y

# Linearizing the EOM:
Y_eom = nonlinear_vmm_equations.Y_eom.subs(
    [
        (Y_nonlin,Y_lin),
        (u,U),
    ]
)

Y_qs_eq = sp.Eq(Y_qs,
                p.Yu*u + p.Yv*v + p.Yr*r + p.Ydelta*delta)

fy_eq = sp.Eq(Y_lin,
             p.Yudot*u.diff() + p.Yvdot*v.diff() +  p.Yrdot*r.diff() + Y_qs)

Y_eq = Y_eom.subs([
    (Y_lin,sp.solve(fy_eq,Y_lin)[0]),
    (Y_qs,sp.solve(Y_qs_eq,Y_qs)[0]),
    ])

## N

# Linearizing the EOM:
N_eom = nonlinear_vmm_equations.N_eom.subs(
    [
        (N_nonlin,N_lin),
        (u,U),
    ]
)

N_qs_eq = sp.Eq(N_qs,
                p.Nu*u + p.Nv*v + p.Nr*r + p.Ndelta*delta)

mz_eq = sp.Eq(N_lin,
             p.Nudot*u.diff() + p.Nvdot*v.diff() + p.Nrdot*r.diff() + N_qs)

N_eq = N_eom.subs([
    (N_lin,sp.solve(mz_eq,N_lin)[0]),
    (N_qs,sp.solve(N_qs_eq,N_qs)[0])
    ])

