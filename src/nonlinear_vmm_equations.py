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

# eq4.24 [1]
fx_eq = sp.Eq(X_nonlin,
    p.Xudot*u.diff() 
)



## Y
# eq4.2 [1]
Y_eom = sp.Eq(m*(v.diff() + r*u + x_G*r.diff()),
             Y_nonlin
             )


## N
# eq4.2 [1]
N_eom = sp.Eq(I_z*r.diff() + m*x_G*(v.diff()+u*r),
             N_nonlin
             )