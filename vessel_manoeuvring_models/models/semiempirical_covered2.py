"""Semi-empirical rudder model as described by Kjellber (2023).
[1] Matusiak, J., 2021. Dynamics of a Rigid Ship - with applications. Aalto University.
[2] Yasukawa, H., Yoshimura, Y., 2015. Introduction of MMG standard method for ship maneuvering predictios. J Mar Sci Technol 20, 37–52. https://doi.org/10.1007/s00773-014-0293-y
[3] Kjellberg, M., Gerhardt, F., Werner, S., n.d. Sailing Performance of Wind-Powered Cargo Vessel in Unsteady Condi- tions.
"""

from typing import Any
import sympy as sp
from sympy import Eq, symbols, Symbol, cos, sin, Derivative, atan, Piecewise, pi
from vessel_manoeuvring_models.symbols import *


# ____________________________________________________________________
# Rudder model

L_R, D_R, C_L, b_R, kappa, C_L_tune, C_D_tune, C_D0_tune = symbols(
    "L_R,D_R,C_L,b_R,kappa,C_L_tune,C_D_tune, C_D0_tune"
)

A_R_C = symbols("A_R_C")  # Rudder area (C)overed by the propeller
A_R_U = symbols("A_R_U")  # Rudder area (U)ncovered by the propeller


V_R_x_C = symbols("V_{R_{Cx}}")
V_R_x_U = symbols("V_{R_{Ux}}")

V_R_y = symbols("V_{R_{y}}")
V_R_C = symbols("V_R_C")  # velocity of the (C)overed part of the rudder
V_R_U = symbols("V_R_U")  # velocity of the (U)ncovered part of the rudder

l_R = sp.symbols("l_R")  # MMG lever arm eq. 24 [2]
V_A = symbols("V_A")
w_f, rho = sp.symbols("w_f,rho,")
x_R, y_R, z_R = sp.symbols("x_R y_R z_R")
V_x_C = sp.symbols("V_x_C")
V_x_U = sp.symbols("V_x_U")

eq_V_R_C = Eq(V_R_C, sp.sqrt(V_R_x_C**2 + V_R_y**2))
eq_V_R_U = Eq(V_R_U, sp.sqrt(V_R_x_U**2 + V_R_y**2))
V_x_corr = symbols("V_{x_{corr}}")
eq_V_R_x_C = Eq(V_R_x_C, V_x_corr + q * z_R - r * y_R)
eq_V_R_x_U = Eq(V_R_x_U, V_x_U + q * z_R - r * y_R)

(
    kappa_v,
    kappa_r,
    kappa_v_pos,
    kappa_r_pos,
    kappa_v_neg,
    kappa_r_neg,
    gamma_g,
) = sp.symbols(
    "kappa_v,kappa_r,kappa_v_pos,kappa_r_pos,kappa_v_neg,kappa_r_neg,gamma_g"
)

eq_gamma_g = Eq(gamma_g, atan((-v - r * x_R + p * z_R) / V_R_x_C))
eq_kappa_v = Eq(kappa_v,Piecewise(
                                (kappa_v_pos, gamma_g > 0),
                                (kappa_v_neg, gamma_g <= 0 ),  
                                  ))
eq_kappa_r = Eq(kappa_r,Piecewise(
                                (kappa_r_pos, gamma_g > 0),
                                (kappa_r_neg, gamma_g <= 0),  
                                  ))

eq_V_R_y = Eq(V_R_y, -kappa_v * v - kappa_r * r * x_R + p * z_R)
eq_V_x_no_propeller = Eq(V_x_U, V_A)
eq_V_A = Eq(V_A, (1 - w_f) * u)

C_D_C = symbols("C_D_C")
C_D_U = symbols("C_D_U")

lambda_R = symbols("lambda_R")  # Limited radius of slipstream (if any)

L_R_C = symbols("L_R_C")
L_R_U = symbols("L_R_U")

# The expressions for rudder lift and drag forces are:
eq_L = Eq(
    L_R,
    L_R_C + L_R_U,
)  # modified eq.46

eq_L_C = Eq(
    L_R_C,
    C_L_tune * 1 / 2 * rho * C_L * lambda_R * A_R_C * V_R_C**2,
)

eq_L_U = Eq(
    L_R_U,
    C_L_tune * 1 / 2 * rho * C_L * A_R_U * V_R_U**2,
)

eq_D = Eq(
    D_R, 1 / 2 * rho * (C_D_C * A_R_C * V_R_C**2 + C_D_U * A_R_U * V_R_U**2)
)  # modified eq.47

C_L_no_stall = symbols("C_{L_{nostall}}")
C_D_C_no_stall = symbols("C_{D_{Cnostall}}")
C_D_U_no_stall = symbols("C_{D_{Unostall}}")

alpha = symbols("alpha")  # angle of attack
C_D_crossflow = symbols("C_D_crossflow")  # crossflow drag coefficient
AR_e = symbols("AR_e")  # Effective aspect ratio

dC_L_dalpha = symbols("dC_L_dalpha")
# When the angle of attack α is below the stall angle αs, the lift and drag coefficients are:
K_gap = symbols(
    "K_gap"
)  # lift diminishing factor for large angles (probably du to gap between rudder and rudder horn)
eq_CL_no_stall = Eq(
    C_L_no_stall,
    K_gap * (dC_L_dalpha * alpha + C_D_crossflow / AR_e * alpha * sp.Abs(alpha)),
)  # eq.48 (|α|<αs)

C_D0_C = symbols("C_D0_C")  # drag coefficient at zero angle of attack
C_D0_U = symbols("C_D0_U")  # drag coefficient at zero angle of attack

e_0 = symbols("e_0")  # Oswald efficiency factor
eq_CD_no_stall_C = Eq(
    C_D_C_no_stall, C_D0_C + C_D_tune * C_L**2 / (pi * AR_e * e_0)
)  # eq.49 (|α|<αs)
eq_CD_no_stall_U = Eq(
    C_D_U_no_stall, C_D0_U + C_D_tune * C_L**2 / (pi * AR_e * e_0)
)  # eq.49 (|α|<αs)

C_F_C = symbols("C_F_C")  # frictional coeffient at zerp angle of attack.
C_F_U = symbols("C_F_U")  # frictional coeffient at zerp angle of attack.
eq_C_D0_C = Eq(C_D0_C, C_D0_tune * 2.5 * C_F_C)
eq_C_D0_U = Eq(C_D0_U, C_D0_tune * 2.5 * C_F_U)

Re_F_C = symbols("Re_F_C")  # Reynold number based on the rudder's mean chord.
Re_F_U = symbols("Re_F_U")  # Reynold number based on the rudder's mean chord.

eq_C_F_C = Eq(C_F_C, 0.075 / (sp.log(Re_F_C - 2, 10) ** 2))
eq_C_F_U = Eq(C_F_U, 0.075 / (sp.log(Re_F_U - 2, 10) ** 2))

nu, c = sp.symbols("nu c")  # kinematic_viscosity, coord length
eq_Re_F_C = Eq(Re_F_C, V_R_C * c / nu)
eq_Re_F_U = Eq(Re_F_U, V_R_U * c / nu)

c_t = symbols("c_t")  # tip chord
c_r = symbols("c_r")  # root chord
eq_c = Eq(c, (c_r + c_t) / 2)

Lambda = symbols("lambda")  # taper ration of the rudder
eq_Lambda = Eq(Lambda, c_t / c_r)  # taper ration of the rudder
eq_C_DC = Eq(C_D_crossflow, 0.1 + 1.6 * Lambda)  # eq.51 square tip

delta, delta_lim = symbols("delta delta_lim")
AR_g = symbols("AR_g")  # Geometric aspect ratio?
# Assuming the gap at the rudder root is small at rudder angle δ = 0 and large at maximum rudder angle δmax,
# the effective aspect ratio of the rudder is taken as:
eq_AR_e = Eq(AR_e, 2 * AR_g)  # eq.52
eq_AR_g = Eq(AR_g, b_R**2 / A_R)
s = symbols("s")
eq_K_gap = Eq(
    K_gap,
    Piecewise((1, sp.Abs(delta) < delta_lim), (1 + s*(sp.Abs(delta)-delta_lim)**2, sp.Abs(delta) >= delta_lim)),
)

# The lift curve slope of the rudder is given by:
a_0 = symbols("a_0")  # section lift curve slope
Omega = symbols("Omega")  # sweep angle of the quarter chord line (see Lewis, 1989)
eq_C_L_alpha = Eq(
    dC_L_dalpha,
    a_0 * AR_e / (1.8 + cos(Omega) * sp.sqrt(AR_e**2 / (cos(Omega) ** 4) + 4)),
)

eq_a0 = Eq(a_0, 0.9 * 2 * pi)  # section lift curve slope


eq_C_L = Eq(
    C_L,
    eq_CL_no_stall.rhs
    )

eq_C_D_C = Eq(
    C_D_C,
            eq_CD_no_stall_C.rhs    
)

eq_C_D_U = Eq(
    C_D_U,
    eq_CD_no_stall_U.rhs
        
)


gamma = symbols("gamma")

gamma_0 = symbols("gamma_0")  # Inflow angle from the hull
eq_alpha = Eq(alpha, delta + gamma_0 + gamma)
V_y = symbols("V_y")
eq_gamma = Eq(gamma, atan(V_R_y / V_R_x_C))


# ____________________________________________________________________
# The effect of propeller action on the rudder flow
V_infty, C_Th, r_0 = sp.symbols("V_\infty,C_Th,r_0")
eq_V_infty = Eq(V_infty, V_A * sp.sqrt(1 + C_Th))
eq_C_Th = Eq(
    C_Th,
    # thrust_propeller / (sp.Rational(1, 2) * rho * V_A**2 * pi * (2 * r_0) ** 2 / 4)/2,  # Why /2????
    thrust_propeller / (sp.Rational(1, 2) * rho * V_A**2 * pi * (2 * r_0) ** 2 / 4),
)
r_infty = sp.symbols("r_\infty")
eq_r_infty = Eq(r_infty, r_0 * sp.sqrt(sp.Rational(1, 2) * (1 + V_A / V_infty)))
r_x, x = sp.symbols("r_x,x")
eq_r = Eq(
    r_x,
    r_0
    * (0.14 * (r_infty / r_0) ** 3 + (r_infty / r_0) * (x / r_0) ** 1.5)
    / (0.14 * (r_infty / r_0) ** 3 + (x / r_0) ** 1.5),
)
eq_V_x_C = Eq(V_x_C, V_infty * (r_infty / r_x) ** 2)
r_Delta = sp.symbols("r_Delta")
Delta_r_x = r_Delta  # (MAK, Matusiak)
eq_r_Delta = Eq(r_Delta, 0.15 * x * ((V_x_C - V_A) / (V_x_C + V_A)))
# eq_V_x_corr = Eq(V_x_corr, (V_x_C - V_A) * r_x / (r_x + r_Delta) + V_A)
eq_V_x_corr = Eq(V_x_corr, (V_x_C - V_A) * (r_x / (r_x + r_Delta)) ** 2 + V_A)

# The limited radius of the slipstream in the lateral direction also diminishes the
# rudder lift force, which can be taken into account by multiplying the lift coefficient
# CL by a factor λ, given by:
f = symbols("f")
d = symbols("d")
c = symbols("c")  # mean chord
eq_lambda_R = Eq(lambda_R, (V_A / V_x_corr) ** f)  # eq.67
# eq_lambda_R = Eq(lambda_R,1)  #eq.67 (This one is a bit strange, perhaps =1 is more reasonable?)
eq_f = Eq(f, 2 * (2 / (2 + d / c)) ** 8)
eq_d = Eq(d, sp.sqrt(pi / 4) * (r_x + Delta_r_x))

# _____________________________________________________________________
# Express Lift and Drag (flow reference frame) in the ship reference frame:
D_F, L_F, alpha_f = sp.symbols(
    "D_F,L_F,alpha_f"
)  # Forces in flow direction (alfa_F=kappa*gamma)
eq_X_R = Eq(X_R, -(-L_R * sin(alpha_f) + D_R * cos(alpha_f)))
eq_Y_R = Eq(Y_R, (L_R * cos(alpha_f) + D_R * sin(alpha_f)))
eq_N_R = Eq(N_R, x_R * Y_R)
# eq_alpha_f = Eq(alpha_f, kappa * gamma)
eq_alpha_f = Eq(
    alpha_f, eq_alpha.rhs.subs(delta, 0)
)  # setting delta=0 puts the rudder and ship in the same orientation.
