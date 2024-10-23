"""Semi-empirical rudder model as described by Kjellber (2023).
[1] Matusiak, J., 2021. Dynamics of a Rigid Ship - with applications. Aalto University.
[2] Yasukawa, H., Yoshimura, Y., 2015. Introduction of MMG standard method for ship maneuvering predictios. J Mar Sci Technol 20, 37–52. https://doi.org/10.1007/s00773-014-0293-y
[3] Kjellberg, M., Gerhardt, F., Werner, S., n.d. Sailing Performance of Wind-Powered Cargo Vessel in Unsteady Condi- tions.
"""

from typing import Any
import sympy as sp
from sympy import Eq, symbols, Symbol, cos, sin, Derivative, atan, Piecewise, pi
from vessel_manoeuvring_models.symbols import *

# _____________________________________________________________________
# Express Lift and Drag (flow reference frame) in the ship reference frame:
D_F, L_F, alpha_f = sp.symbols(
    "D_F,L_F,alpha_f"
)  # Forces in flow direction (alfa_F=kappa*gamma)
L_R, D_R, C_L, b_R, kappa, C_L_tune, C_D_tune, C_D0_tune = symbols(
    "L_R,D_R,C_L,b_R,kappa,C_L_tune,C_D_tune, C_D0_tune"
)
eq_X_R = sp.Eq(X_R, -(-L_R * sin(alpha_f) + D_R * cos(alpha_f)))
eq_Y_R = sp.Eq(Y_R, (L_R * cos(alpha_f) + D_R * sin(alpha_f)))
x_R, y_R, z_R = sp.symbols("x_R y_R z_R")
eq_N_R = sp.Eq(N_R, x_R * Y_R)


# _____________________________________________________________________
# Lift
V_R = symbols("V_R")  # velocity at the the rudder
eq_L = Eq(
    L_R,
    C_L_tune * 1 / 2 * rho * C_L * A_R * V_R**2
)

C_D_crossflow = symbols("C_D_crossflow")  # crossflow drag coefficient
AR_e = symbols("AR_e")  # Effective aspect ratio
alpha = symbols("alpha")  # angle of attack
K_gap = symbols(
    "K_gap"
)  # lift diminishing factor for large angles (probably du to gap between rudder and rudder horn)
dC_L_dalpha = symbols("dC_L_dalpha")

eq_CL_no_stall = Eq(
    C_L,
    K_gap * (dC_L_dalpha * alpha + C_D_crossflow / AR_e * alpha * sp.Abs(alpha)),
)  # eq.48 (|α|<αs)

a_0 = symbols("a_0")  # section lift curve slope
Omega = symbols("Omega")  # sweep angle of the quarter chord line (see Lewis, 1989)
eq_C_L_alpha = Eq(
    dC_L_dalpha,
    a_0 * AR_e / (1.8 + cos(Omega) * sp.sqrt(AR_e**2 / (cos(Omega) ** 4) + 4)),
)
eq_a0 = Eq(a_0, 0.9 * 2 * pi)  # section lift curve slope

s = symbols("s")
delta, delta_lim = symbols("delta delta_lim")
eq_K_gap = Eq(
    K_gap,
    Piecewise((1, sp.Abs(delta) < delta_lim), (1 + s*(sp.Abs(delta)-delta_lim)**2, sp.Abs(delta) >= delta_lim)),
)


Lambda = symbols("lambda")  # taper ration of the rudder
eq_C_D = Eq(C_D_crossflow, 0.1 + 1.6 * Lambda)  # eq.51 square tip
c_t = symbols("c_t")  # tip chord
c_r = symbols("c_r")  # root chord
eq_Lambda = Eq(Lambda, c_t / c_r)  # taper ration of the rudder

AR_g = symbols("AR_g")  # Geometric aspect ratio?
eq_AR_e = Eq(AR_e, 2 * AR_g)  # eq.52
eq_AR_g = Eq(AR_g, b_R**2 / A_R)

# Rudder velocity
V_R_y = symbols("V_{R_{y}}")
V_R_x = symbols("V_{R_{x}}")
eq_V_R = sp.Eq(V_R, sp.sqrt(V_R_x**2 + V_R_y**2))
#eq_V_R_x = sp.Eq(V_R_x, V_x + q * z_R - r * y_R)
#eq_V_R_y = sp.Eq(V_R_y, -kappa_v_tot * v - kappa_r_tot * r * x_R + p * z_R)

alpha = symbols("alpha")  # angle of attack
gamma_0 = symbols("gamma_0")  # Inflow angle from the hull
gamma = symbols("gamma")
eq_alpha = Eq(alpha, delta + gamma_0 + gamma)
V_y = symbols("V_y")
eq_gamma = Eq(gamma, atan(V_R_y / V_R_x))
eq_alpha_f = Eq(
    alpha_f, eq_alpha.rhs.subs(delta, 0)
)  # setting delta=0 puts the rudder and ship in the same orientation.


#
# Drag
C_D = symbols("C_D")
eq_D = Eq(
    D_R, 1 / 2 * rho * (C_D * A_R * V_R**2)
)

e_0 = symbols("e_0")  # Oswald efficiency factor
C_D0 = symbols("C_D0")  # drag coefficient at zero angle of attack
eq_CD_no_stall = Eq(
    C_D, C_D0 + C_D_tune * C_L**2 / (pi * AR_e * e_0)
)  # eq.49 (|α|<αs)

C_F = symbols("C_F")  # frictional coeffient at zerp angle of attack.
eq_C_D0 = Eq(C_D0, C_D0_tune * 2.5 * C_F)
Re_F = symbols("Re_F")  # Reynold number based on the rudder's mean chord.
eq_C_F = Eq(C_F, 0.075 / (sp.log(Re_F - 2, 10) ** 2))
nu, c = sp.symbols("nu c")  # kinematic_viscosity, coord length
eq_Re_F = sp.Eq(Re_F, V_R * c / nu)
eq_c = sp.Eq(c, (c_r + c_t) / 2)