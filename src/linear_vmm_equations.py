"""
References:
[1] : Matusiak, Jerzy. Dynamics of a Rigid Ship - with Applications, 3rd Edition, 2021.
[2] : Triantafyllou, Michael S, and Franz S Hover. “MANEUVERING AND CONTROL OF MARINE VEHICLES.” Massachusetts Institute of Technology, 2003, 152.
"""


import sympy as sp
from src.symbols import *
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

fx_eq = sp.Eq(X_lin,
             p.Xudot*u.diff() + p.Xu*u + p.Xvdot*v.diff() + p.Xv*v + p.Xrdot*r.diff() + p.Xr*r + p.Xdelta*delta)

X_eq = X_eom.subs(X_lin,sp.solve(fx_eq,X_lin)[0])

## Y

# Linearizing the EOM:
Y_eom = nonlinear_vmm_equations.Y_eom.subs(
    [
        (Y_nonlin,Y_lin),
        (u,U),
    ]
)


fy_eq = sp.Eq(Y_lin,
             p.Yudot*u.diff() + p.Yu*u + p.Yvdot*v.diff() + p.Yv*v + p.Yrdot*r.diff() + p.Yr*r + p.Ydelta*delta)

Y_eq = Y_eom.subs(Y_lin,sp.solve(fy_eq,Y_lin)[0])

## N

# Linearizing the EOM:
N_eom = nonlinear_vmm_equations.N_eom.subs(
    [
        (N_nonlin,N_lin),
        (u,U),
    ]
)

mz_eq = sp.Eq(N_lin,
             p.Nudot*u.diff() + p.Nu*u + p.Nvdot*v.diff() + p.Nv*v + p.Nrdot*r.diff() + p.Nr*r + p.Ndelta*delta)

N_eq = N_eom.subs(N_lin,sp.solve(mz_eq,N_lin)[0])
