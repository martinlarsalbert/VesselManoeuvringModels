import sympy as sp
from src.symbols import *
import pandas as pd

p = df_parameters['symbol']

## X
X_eq = sp.Eq(
    (p.Xudot-m)*u.diff() + p.Xu*u + p.Xvdot*v.diff() + p.Xv*v + p.Xrdot*r.diff() + p.Xr*r + p.Xdelta*delta,
    0
)

## Y
Y_eq = sp.Eq(
    p.Yudot*u.diff() + p.Yu*u + (p.Yvdot-m)*v.diff() + p.Yv*v + (p.Yrdot-m*x_G)*r.diff() + (p.Yr-m*U)*r + p.Ydelta*delta,
    0
)

## N
N_eq = sp.Eq(
    p.Nudot*u.diff() + p.Nu*u + (p.Nvdot-m*x_G)*v.diff() + p.Nv*v + (p.Nrdot-I_z)*r.diff() + (p.Nr-m*x_G*U)*r + p.Ndelta*delta,
    0
)