"""
[1] : Yasukawa, H., Yoshimura, Y., 2015. Introduction of MMG standard method for ship maneuvering predictions. J Mar Sci Technol 20, 37–52. https://doi.org/10.1007/s00773-014-0293-y
"""

import sympy as sp
from vessel_manoeuvring_models.symbols import *
import pandas as pd
from vessel_manoeuvring_models.nonlinear_vmm_equations import *
from vessel_manoeuvring_models.models.vmm import VMM
from vessel_manoeuvring_models.models.MMG_propeller import *
from vessel_manoeuvring_models.models.MMG_rudder import *
from vessel_manoeuvring_models.models.MMG_hull import *

p = df_parameters["symbol"]

subs = [
    (p.Xvdot, 0),
    (p.Xrdot, 0),
    (p.Yudot, 0),
    (p.Nudot, 0),
]

## X

X_qs_eq_raw = sp.Eq(X_D, X_H + X_R + X_P)
X_qs_eq = sp.Eq(X_D, eq_X_H.rhs + X_R_solution.rhs + X_P_solution[0][X_P])

fx_eq = fx_eq.subs(subs)
X_eq = X_eom.subs(
    [(X_force, sp.solve(fx_eq, X_force)[0]), (X_D, sp.solve(X_qs_eq, X_D)[0])]
)

## Y
Y_qs_eq_raw = sp.Eq(Y_D, Y_H + Y_R)
Y_qs_eq = sp.Eq(Y_D, eq_Y_H.rhs + Y_R_solution.rhs)

fy_eq = fy_eq.subs(subs)
Y_eq = Y_eom.subs(
    [
        (Y_force, sp.solve(fy_eq, Y_force)[0]),
        (Y_D, sp.solve(Y_qs_eq, Y_D)[0]),
    ]
)

## N
N_qs_eq_raw = sp.Eq(N_D, N_H + N_R)
N_qs_eq = sp.Eq(N_D, eq_N_H.rhs + N_R_solution.rhs)

mz_eq = mz_eq.subs(subs)
N_eq = N_eom.subs(
    [
        (N_force, sp.solve(mz_eq, N_force)[0]),
        (N_D, sp.solve(N_qs_eq, N_D)[0]),
    ]
)

MMG_model = VMM(X_eq=X_eq, Y_eq=Y_eq, N_eq=N_eq)
