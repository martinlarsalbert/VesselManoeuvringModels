"""
References:
[1] : Matusiak, Jerzy. Dynamics of a Rigid Ship - with Applications, 3rd Edition, 2021.
[2] : Triantafyllou, Michael S, and Franz S Hover. “MANEUVERING AND CONTROL OF MARINE VEHICLES.” Massachusetts Institute of Technology, 2003, 152.
"""


import sympy as sp
from src.symbols import *
import pandas as pd
from src.nonlinear_vmm_equations import *

p = df_parameters['symbol']

## X

X_qs_eq = sp.Eq(X_qs,
        p.Xu*u + p.Xvv*v**2 + p.Xdeltadelta*delta**2 + 
        p.Xvr*v*r)

X_eq = X_eom.subs([
    (X_nonlin,sp.solve(fx_eq,X_nonlin)[0]),
    (X_qs,sp.solve(X_qs_eq,X_qs)[0])
])

## Y

Y_qs_eq = sp.Eq(Y_qs,
        p.Yv*v + p.Yr*r + p.Yu*u + p.Yvv*v*sp.Abs(v) + 
        p.Ydelta*delta + p.Yudelta*u*delta
        
    )

Y_eq = Y_eom.subs([
    (Y_nonlin,sp.solve(fy_eq,Y_nonlin)[0]),
    (Y_qs,sp.solve(Y_qs_eq,Y_qs)[0]),
    ])

## N

N_qs_eq = sp.Eq(N_qs,
        p.Nv*v + p.Nr*r + p.Nu*u + p.Nrr*r*sp.Abs(r) + 
        p.Ndelta*delta + p.Nudelta*u*delta
    )

N_eq = N_eom.subs([
    (N_nonlin,sp.solve(mz_eq,N_nonlin)[0]),
    (N_qs,sp.solve(N_qs_eq,N_qs)[0]),
])
