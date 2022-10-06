"""
[1] : Yasukawa, H., Yoshimura, Y., 2015. Introduction of MMG standard method for ship maneuvering predictions. J Mar Sci Technol 20, 37–52. https://doi.org/10.1007/s00773-014-0293-y
"""

import sympy as sp
from vessel_manoeuvring_models.symbols import *
import pandas as pd
from vessel_manoeuvring_models.nonlinear_vmm_equations import *

p = df_parameters["symbol"]

eq_X_H = sp.Eq(
    X_H, -R0 + p.Xvv * v ** 2 + p.Xvr * v * r + p.Xrr * r ** 2 + p.Xvvvv * v ** 4
)

eq_Y_H = sp.Eq(
    Y_H,
    p.Yv * v
    + p.Yr * r
    + p.Yvvv * v ** 3
    + p.Yvvr * v ** 2 * r
    + p.Yvrr * v * r ** 2
    + p.Yrrr * r ** 3,
)

eq_N_H = sp.Eq(
    N_H,
    p.Nv * v
    + p.Nr * r
    + p.Nvvv * v ** 3
    + p.Nvvr * v ** 2 * r
    + p.Nvrr * v * r ** 2
    + p.Nrrr * r ** 3,
)
