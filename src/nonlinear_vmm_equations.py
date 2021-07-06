"""
References:
[1] : Matusiak, Jerzy. Dynamics of a Rigid Ship - with Applications, 3rd Edition, 2021.
[2] : Triantafyllou, Michael S, and Franz S Hover. “MANEUVERING AND CONTROL OF MARINE VEHICLES.” Massachusetts Institute of Technology, 2003, 152.
"""


import sympy as sp
from src.symbols import *
import pandas as pd

p = df_parameters['symbol']

## X
# eq4.2 [1]
X_eom = sp.Eq(m*(u.diff()-r*v-x_G*r**2),
             X_nonlin
             )


X_qs_eq = sp.Eq(X_qs,
        p.Xu*u + p.Xv*v + p.Xr*r + p.Xdelta*delta
    )

# eq4.24 [1]
fx_eq = sp.Eq(X_nonlin,
    p.Xudot*u.diff() + X_qs
)

X_eq = X_eom.subs(X_nonlin,sp.solve(fx_eq,X_nonlin)[0])

## Y
# eq4.2 [1]
Y_eom = sp.Eq(m*(v.diff() + r*u + x_G*r.diff()),
             Y_nonlin
             )

Y_qs_eq = sp.Eq(Y_qs,
        p.Yu*u + p.Yv*v + p.Yr*r + p.Ydelta*delta
    )

fy_eq = sp.Eq(Y_nonlin,
             p.Yvdot*v.diff()  + Y_qs + p.Yrdot*r.diff()
)            

Y_eq = Y_eom.subs(Y_nonlin,sp.solve(fy_eq,Y_nonlin)[0])

## N
# eq4.2 [1]
N_eom = sp.Eq(I_z*r.diff() + m*x_G*(v.diff()+u*r),
             N_nonlin
             )

N_qs_eq = sp.Eq(N_qs,
        p.Nu*u + p.Nv*v + p.Nr*r + p.Ndelta*delta
    )

mz_eq = sp.Eq(N_nonlin,
             p.Nrdot*r.diff()  + N_qs + p.Nvdot*v.diff()
)

N_eq = N_eom.subs(N_nonlin,sp.solve(mz_eq,N_nonlin)[0])
