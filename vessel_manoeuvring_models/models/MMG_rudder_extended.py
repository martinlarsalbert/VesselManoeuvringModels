"""
[2] Fujii H, Tuda T (1961) Experimental research on rudder performance (2). J Soc Naval Archit Jpn 110:31–42 (in Japanese)
"""

import sympy as sp
from vessel_manoeuvring_models.symbols import *
from sympy import Eq, symbols, pi, Piecewise

V_A = symbols("V_A")
C_Th, r_0 = sp.symbols("C_Th,r_0")
gamma_0 = symbols("gamma_0")  # Inflow angle from the hull
gamma_R_neg,gamma_R_pos = symbols("gamma_R_neg,gamma_R_pos")
gamma_R2_neg,gamma_R2_pos = symbols("gamma_R2_neg,gamma_R2_pos")

#gamma_0_neg,gamma_0_pos = symbols("gamma_0_neg,gamma_0_pos")
beta_R = symbols("beta_R")

eq_F_N = Eq(
    F_N, sp.Rational(1, 2) * rho * A_R * U_R ** 2 * f_alpha * sp.sin(-alpha_R)
)
eq_U_R = Eq(U_R, sp.sqrt(u_R ** 2 + v_R ** 2))
eq_alpha_R = Eq(
    alpha_R, delta + sp.atan(v_R / u_R) + gamma_0
)  # (-delta other coordinate def.)

#eq_beta_R = Eq(
#    beta_R, -delta - sp.atan(v_R / u_R)
#)  # (-delta other coordinate def.)

eq_v_R = Eq(v_R, U * gamma_R * beta_R)
#eq_beta_R = Eq(beta_R, beta - l_R' * r')
eq_beta_R = Eq(beta_R, beta - l_R/L * r/(U/L))
eq_beta = Eq(beta,sp.atan2(-v,u))

eq_gamma_R = Eq(gamma_R, Piecewise((gamma_R_neg+gamma_R2_neg*sp.Abs(beta_R),beta_R<=0),
                                   (gamma_R_pos+gamma_R2_pos*sp.Abs(beta_R),beta_R>0)))



eq_u_R = Eq(
    u_R,
    epsilon
    * u
    * (1 - w_p)
    * sp.sqrt(eta * (1 + kappa * (sp.sqrt(1 + C_Th) - 1)) ** 2 + (1 - eta)),
)

eq_C_Th = Eq(
    C_Th,
    thrust_propeller / (sp.Rational(1, 2) * rho * V_A**2 * pi * (2 * r_0) ** 2 / 4),
)

eq_V_A = Eq(V_A, (1 - w_p) * u)

eq_eta = Eq(eta, D / H_R)  # ratio of propeller doameter to rudder span

eq_Lambda = Eq(Lambda, H_R / C_R)  # Aspect ratio

eq_f_alpha = Eq(f_alpha, 6.13 * Lambda / (Lambda + 2.25))  # [2]

## Projections:
eq_X_R = Eq(
    X_R, -(1 - t_R) * F_N * sp.sin(-delta)
)  # delta minus due to different coordinatesystem

#eq_Y_R = Eq(Y_R, -(1 + a_H) * F_N * sp.cos(-delta))
eq_Y_R = Eq(Y_R, - F_N * sp.cos(-delta))
#eq_N_R = Eq(N_R, -(x_r + a_H * x_H) * F_N * sp.cos(-delta))
eq_N_R = Eq(N_R, x_r*Y_R)
