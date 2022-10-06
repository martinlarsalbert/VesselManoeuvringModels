"""
References:
[1] : Matusiak, Jerzy. Dynamics of a Rigid Ship - with Applications, 3rd Edition, 2021.
[2] : Triantafyllou, Michael S, and Franz S Hover. “MANEUVERING AND CONTROL OF MARINE VEHICLES.” Massachusetts Institute of Technology, 2003, 152.
"""


import sympy as sp
from vessel_manoeuvring_models.symbols import *
import vessel_manoeuvring_models.linear_vmm_equations as linear_vmm_equations

import pandas as pd

p = df_parameters['symbol']

#Apply the following simplifications:
subs = [
    (x_G,0),
    (p.Xvdot,0),
    (p.Xrdot,0),
    (p.Yudot,0),
    (p.Yrdot,0),
    (p.Nudot,0),
    (p.Nvdot,0),    
]

## Simplify


## X
X_eom = linear_vmm_equations.X_eom.subs(subs)
#X_qs_eq = linear_vmm_equations.X_qs_eq.subs(subs)
fx_eq = linear_vmm_equations.fx_eq.subs(subs)
X_eq = linear_vmm_equations.X_eq.subs(subs)

## Y
Y_eom = linear_vmm_equations.Y_eom.subs(subs)
#Y_qs_eq = linear_vmm_equations.Y_qs_eq.subs(subs)
fy_eq = linear_vmm_equations.fy_eq.subs(subs)
Y_eq = linear_vmm_equations.Y_eq.subs(subs)

## N
N_eom = linear_vmm_equations.N_eom.subs(subs)
#N_qs_eq = linear_vmm_equations.N_qs_eq.subs(subs)
mz_eq = linear_vmm_equations.mz_eq.subs(subs)
N_eq = linear_vmm_equations.N_eq.subs(subs)
