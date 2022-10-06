"""
[2] Fujii H, Tuda T (1961) Experimental research on rudder performance (2). J Soc Naval Archit Jpn 110:31–42 (in Japanese)
"""

import sympy as sp
from vessel_manoeuvring_models.symbols import *

eq_F_N = sp.Eq(
    F_N, sp.Rational(1, 2) * rho * A_R * U_R ** 2 * f_alpha * sp.sin(alpha_R)
)
eq_U_R = sp.Eq(U_R, sp.sqrt(u_R ** 2 + v_R ** 2))
eq_alpha_R = sp.Eq(
    alpha_R, -delta - sp.atan(v_R / u_R)
)  # (-delta other coordinate def.)
eq_v_R = sp.Eq(v_R, U * gamma_R * beta_R)
eq_beta_R = sp.Eq(beta_R, beta - l_R * r)

eq_u_R = sp.Eq(
    u_R,
    epsilon
    * u
    * (1 - w_p)
    * sp.sqrt(eta * (1 + kappa * (sp.sqrt(1 + C_Th) - 1)) ** 2 + (1 - eta)),
)

eq_eta = sp.Eq(eta, D / H_R)

eq_Lambda = sp.Eq(Lambda, H_R / C_R)  # Aspect ratio

eq_f_alpha = sp.Eq(f_alpha, 6.13 * Lambda / (Lambda + 2.25))  # [2]

## Projections:
eq_X_R = sp.Eq(
    X_R, -(1 - t_R) * F_N * sp.sin(-delta)
)  # delta minus due to different coordinatesystem

eq_Y_R = sp.Eq(Y_R, -(1 + a_H) * F_N * sp.cos(-delta))

eq_N_R = sp.Eq(N_R, -(x_r + a_H * x_H) * F_N * sp.cos(-delta))

## Solve:
eqs = [
    eq_eta,
    eq_u_R,
    eq_v_R,
    eq_alpha_R,
    eq_U_R,
    eq_f_alpha,
    eq_F_N,
]

solution = sp.solve(
    eqs,
    eta,
    u_R,
    v_R,
    alpha_R,
    U_R,
    f_alpha,
    F_N,
    dict=True,
    simplify=False,
    rational=False,
)

# F_N_solution = sp.simplify(solution[0][F_N])
F_N_solution = solution[0][F_N]
X_R_solution = eq_X_R.subs(F_N, F_N_solution)
Y_R_solution = eq_Y_R.subs(F_N, F_N_solution)
N_R_solution = eq_N_R.subs(F_N, F_N_solution)
