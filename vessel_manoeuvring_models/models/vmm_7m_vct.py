import sympy as sp
from vessel_manoeuvring_models.parameters import df_parameters
from vessel_manoeuvring_models.symbols import *

p = df_parameters["symbol"]


eq_X_H = sp.Eq(X_H, p.X0 + p.Xu * u + p.Xvv * v**2 + p.Xrr * r**2 + p.Xvr * v * r)

eq_Y_H = sp.Eq(
    Y_H,
    # p.Y0+
    +p.Yv * v + p.Yvvv * v**3
    # +p.Yr*r
    + p.Yvrr * v * r**2,
)

eq_N_H = sp.Eq(
    N_H,
    # p.N0
    +p.Nv * v + p.Nvvv * v**3 + p.Nr * r + p.Nrrr * r**3 + p.Nvvr * v**2 * r,
)
