"""
References:
[1] : Matusiak, Jerzy. Dynamics of a Rigid Ship - with Applications, 3rd Edition, 2021.
[2] : Triantafyllou, Michael S, and Franz S Hover. “MANEUVERING AND CONTROL OF MARINE VEHICLES.” Massachusetts Institute of Technology, 2003, 152.
"""


import sympy as sp
from vessel_manoeuvring_models.symbols import *
from vessel_manoeuvring_models.parameters import *

p = df_parameters["symbol"]

## X
# eq4.2 [1]
X_eom = sp.Eq(m * (u1d - r * v - x_G * r ** 2), X_force)

# eq4.24 [1]
X_qs_eq = sp.Eq(
    X_D,
    p.Xu * u
    + p.Xuu * u ** 2
    + p.Xuuu * u ** 3
    + p.Xvv * v ** 2
    + p.Xrr * r ** 2
    + p.Xdeltadelta * delta ** 2
    + p.Xvr * v * r
    + p.Xvdelta * v * delta
    + p.Xrdelta * r * delta
    + p.Xuvv * u * v ** 2
    + p.Xurr * u * r ** 2
    + p.Xudeltadelta * u * delta ** 2
    + p.Xurdelta * u * r * delta
    + p.Xuvr * u * v * r
    + p.Xuvdelta * u * v * delta
    + p.Xvrdelta * v * r * delta
    + p.Xthrust * thrust,
)

# eq4.24 [1]
fx_eq = sp.Eq(X_force, p.Xudot * u1d + p.Xvdot * v1d + p.Xrdot * r1d + X_D)

X_eq = X_eom.subs(
    [(X_force, sp.solve(fx_eq, X_force)[0]), (X_D, sp.solve(X_qs_eq, X_D)[0])]
)

## Y
# eq4.2 [1]
Y_eom = sp.Eq(m * (v1d + r * u + x_G * r1d), Y_force)

# eq4.26 [1]
Y_qs_eq = sp.Eq(
    Y_D,
    p.Yuu * u ** 2
    + p.Yv * v
    + p.Yr * r
    + p.Ydelta * delta
    + p.Yudelta * u * delta
    + p.Yuv * u * v
    + p.Yur * u * r
    + p.Yuuv * u ** 2 * v
    + p.Yuur * u ** 2 * r
    + p.Yuudelta * u ** 2 * delta
    + p.Yvvv * v ** 3
    + p.Yrrr * r ** 3
    + p.Yrrdelta * r ** 2 * delta
    + p.Yvrr * v * r ** 2
    + p.Yvvr * v ** 2 * r
    + p.Yvvdelta * v ** 2 * delta
    + p.Yvrdelta * v * r * delta
    + p.Yrdeltadelta * r * delta ** 2
    + p.Yvdeltadelta * v * delta ** 2,
)

fy_eq = sp.Eq(Y_force, p.Yvdot * v1d + p.Yrdot * r1d + p.Yudot * u1d + Y_D)

Y_eq = Y_eom.subs(
    [
        (Y_force, sp.solve(fy_eq, Y_force)[0]),
        (Y_D, sp.solve(Y_qs_eq, Y_D)[0]),
    ]
)

## N
# eq4.2 [1]
N_eom = sp.Eq(I_z * r1d + m * x_G * (v1d + u * r), N_force)

# eq.4.27 [1]
N_qs_eq = sp.Eq(
    N_D,
    p.Nuu * u ** 2
    + p.Nv * v
    + p.Nr * r
    + p.Ndelta * delta
    + p.Nudelta * u * delta
    + p.Nuv * u * v
    + p.Nur * u * r
    + p.Nuuv * u ** 2 * v
    + p.Nuur * u ** 2 * r
    + p.Nuudelta * u ** 2 * delta
    + p.Nvvv * v ** 3
    + p.Nrrr * r ** 3
    + p.Ndeltadeltadelta * delta ** 3
    + p.Nrrdelta * r ** 2 * delta
    + p.Nvrr * v * r ** 2
    + p.Nvvr * v ** 2 * r
    + p.Nvvdelta * v ** 2 * delta
    + p.Nvrdelta * v * r * delta
    + p.Nrdeltadelta * r * delta ** 2
    + p.Nvdeltadelta * v * delta ** 2,
)

mz_eq = sp.Eq(N_force, p.Nrdot * r1d + p.Nvdot * v1d + p.Nudot * v1d + N_D)

N_eq = N_eom.subs(
    [
        (N_force, sp.solve(mz_eq, N_force)[0]),
        (N_D, sp.solve(N_qs_eq, N_D)[0]),
    ]
)
