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

eq_V_R_C = sp.Eq(V_R_C, sp.sqrt(V_R_x_C**2 + V_R_y**2))
eq_V_R_U = sp.Eq(V_R_U, sp.sqrt(V_R_x_U**2 + V_R_y**2))

eq_V_R_x_C = sp.Eq(V_R_x_C, V_x_C + q * z_R - r * y_R)
eq_V_R_x_U = sp.Eq(V_R_x_U, V_x_U + q * z_R - r * y_R)

kappa_v, kappa_r,kappa_v_gamma_g, kappa_r_gamma_g, kappa_v_tot, kappa_r_tot, gamma_g  = sp.symbols("kappa_v,kappa_r,kappa_v_gamma_g, kappa_r_gamma_g, kappa_v_tot, kappa_r_tot, gamma_g")

eq_gamma_g = sp.Eq(gamma_g, atan((-v - r * x_R + p * z_R) / V_R_x_C))
eq_kappa_v_tot = sp.Eq(kappa_v_tot,kappa_v+kappa_v_gamma_g*sp.Abs(gamma_g))
eq_kappa_r_tot = sp.Eq(kappa_r_tot,kappa_r+kappa_r_gamma_g*sp.Abs(gamma_g))

eq_V_R_y = sp.Eq(V_R_y, -kappa_v_tot*v - kappa_r_tot*r * x_R + p * z_R)
eq_V_x_no_propeller = Eq(V_x_U, V_A)
eq_V_A = sp.Eq(V_A, (1 - w_f) * u)

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
eq_CL_no_stall = Eq(
    C_L_no_stall,
    dC_L_dalpha * alpha + C_D_crossflow / AR_e * alpha * sp.Abs(alpha),
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
eq_Re_F_C = sp.Eq(Re_F_C, V_R_C * c / nu)
eq_Re_F_U = sp.Eq(Re_F_U, V_R_U * c / nu)

c_t = symbols("c_t")  # tip chord
c_r = symbols("c_r")  # root chord
eq_c = sp.Eq(c, (c_r + c_t) / 2)

Lambda = symbols("lambda")  # taper ration of the rudder
eq_Lambda = Eq(Lambda, c_t / c_r)  # taper ration of the rudder
eq_C_DC = Eq(C_D_crossflow, 0.1 + 1.6 * Lambda)  # eq.51 square tip

delta, delta_lim = symbols("delta delta_lim")
AR_g = symbols("AR_g")  # Geometric aspect ratio?
# Assuming the gap at the rudder root is small at rudder angle δ = 0 and large at maximum rudder angle δmax,
# the effective aspect ratio of the rudder is taken as:
eq_AR_e = Eq(AR_e, (2 - sp.Abs(delta / delta_lim)) * AR_g)  # eq.52
eq_AR_g = Eq(AR_g, b_R**2 / A_R)

# The lift curve slope of the rudder is given by:
a_0 = symbols("a_0")  # section lift curve slope
Omega = symbols("Omega")  # sweep angle of the quarter chord line (see Lewis, 1989)
eq_C_L_alpha = Eq(
    dC_L_dalpha,
    a_0 * AR_e / (1.8 + cos(Omega) * sp.sqrt(AR_e**2 / (cos(Omega) ** 4) + 4)),
)

eq_a0 = Eq(a_0, 0.9 * 2 * pi)  # section lift curve slope

# Following Lloyd (1989), the stall angle is given by:
alpha_s = symbols("alpha_s")
delta_alpha_s = Symbol("\\Delta \\alpha_s")
# eq_alpha_s = Eq(alpha_s, sp.Rational(1.225) - sp.Rational(0.445)*AR_e + sp.Rational(0.075)*AR_e**2)  #AR_e < 3.0
# eq_alpha_s = Eq(alpha_s, sp.Rational(0.565))  #AR_e > 3.0
eq_alpha_s = Eq(
    alpha_s,
    delta_alpha_s
    + Piecewise(
        (1.225 - 0.445 * AR_e + 0.075 * AR_e**2, AR_e <= 3), (0.565, AR_e > 3)
    ),
)
C_N = symbols("C_N")  # normal force coefficient for a stalling wing
eq_C_N = Eq(C_N, 1.8)  # recommended by Hoerner and Borst (1975)
B_s, B_0 = symbols("B_s,B_0")
C_L_max = sp.Symbol("C_{L_{max}}")
C_D_max_C = sp.Symbol("C_{D_{maxC}}")
C_D_max_U = sp.Symbol("C_{D_{maxU}}")
C_L_stall = symbols("C_{L_{stall}}")
C_D_stall_C = symbols("C_{D_{stallC}}")
C_D_stall_U = symbols("C_{D_{stallU}}")

eq_CL_stall = Eq(
    C_L_stall, C_N * sin(alpha) * cos(alpha) * B_s + sp.sign(alpha)*(C_L_max * B_0)
)  # eq.55 (|α|>=αs)
eq_CD_stall_C = Eq(
    C_D_stall_C, C_N * sin(alpha) ** 2 * B_s + C_D_max_C * B_0
)  # eq.56 (|α|>=αs)
eq_CD_stall_U = Eq(
    C_D_stall_U, C_N * sin(alpha) ** 2 * B_s + C_D_max_U * B_0
)  # eq.56 (|α|>=αs)

u_s = symbols("u_s")
eq_B_s = Eq(
    B_s,
    Piecewise(
        (1, sp.Abs(alpha) > 1.25 * alpha_s),
        ((3 * u_s - 2 * u_s**2) * u_s, sp.Abs(alpha) <= 1.25 * alpha_s),
    ),
)
eq_u_s = Eq(u_s, 4 * (sp.Abs(alpha) - alpha_s) / alpha_s)

eq_B_0 = Eq(B_0, 1 - B_s)
eq_CL_max = Eq(
    C_L_max,
    dC_L_dalpha * alpha_s + C_D_crossflow / AR_e * alpha_s * sp.Abs(alpha_s),
)  # eq.58
eq_CD_max_C = Eq(C_D_max_C, C_D0_C + C_D_tune * C_L_max**2 / (pi * AR_e * e_0))
eq_CD_max_U = Eq(C_D_max_U, C_D0_U + C_D_tune * C_L_max**2 / (pi * AR_e * e_0))


eq_C_L = Eq(
    C_L,
    Piecewise(
        (eq_CL_no_stall.rhs, sp.Abs(alpha) < alpha_s),
        (eq_CL_stall.rhs, sp.Abs(alpha) >= alpha_s),
    ),
)

eq_C_D_C = Eq(
    C_D_C,
    Piecewise(
        (eq_CD_no_stall_C.rhs, sp.Abs(alpha) < alpha_s),
        (eq_CD_no_stall_C.rhs, sp.Abs(alpha) >= alpha_s),
    ),
)

eq_C_D_U = Eq(
    C_D_U,
    Piecewise(
        (eq_CD_no_stall_U.rhs, sp.Abs(alpha) < alpha_s),
        (eq_CD_no_stall_U.rhs, sp.Abs(alpha) >= alpha_s),
    ),
)

gamma = symbols("gamma")
kappa = symbols("kappa")
kappa_outer, kappa_inner = symbols("kappa_outer, kappa_inner")

kappa_gamma = symbols("kappa_gamma")

gamma_0 = symbols("gamma_0")  # Inflow angle from the hull
eq_alpha = Eq(
    alpha, delta + gamma_0 + gamma
)
V_y = symbols("V_y")
eq_gamma = Eq(gamma, atan(V_R_y / V_R_x_C))

#eq_kappa = Eq(
#    kappa,
#    Piecewise(
#        (kappa_outer, ((gamma >= 0) & (y_R <= 0)) | ((gamma < 0) & (y_R > 0))),
#        (kappa_inner, ((gamma >= 0) & (y_R > 0)) | ((gamma < 0) & (y_R <= 0))),
#    ),
#)

# ____________________________________________________________________
# The effect of propeller action on the rudder flow
V_x_corr = symbols("V_{x_{corr}}")
V_infty, C_Th, r_0 = sp.symbols("V_\infty,C_Th,r_0")
eq_V_infty = sp.Eq(V_infty, V_A * sp.sqrt(1 + C_Th))
eq_C_Th = sp.Eq(
    C_Th,
    #thrust_propeller / (sp.Rational(1, 2) * rho * V_A**2 * pi * (2 * r_0) ** 2 / 4)/2,  # Why /2????
    thrust_propeller / (sp.Rational(1, 2) * rho * V_A**2 * pi * (2 * r_0) ** 2 / 4), 
)
r_infty = sp.symbols("r_\infty")
eq_r_infty = sp.Eq(r_infty, r_0 * sp.sqrt(sp.Rational(1, 2) * (1 + V_A / V_infty)))
r_p, x = sp.symbols("r_p,x")
r_x = r_p  # (r_x MAK, r_p matusiak)
eq_r = sp.Eq(
    r_p,
    r_0
    * (0.14 * (r_infty / r_0) ** 3 + (r_infty / r_0) * (x / r_0) ** 1.5)
    / (0.14 * (r_infty / r_0) ** 3 + (x / r_0) ** 1.5),
)
eq_V_x_C = sp.Eq(V_x_C, V_infty * (r_infty / r_p) ** 2)
r_Delta = sp.symbols("r_Delta")
Delta_r_x = r_Delta  # (MAK, Matusiak)
eq_r_Delta = sp.Eq(r_Delta, 0.15 * x * ((V_x_C - V_A) / (V_x_C + V_A)))
#eq_V_x_corr = sp.Eq(V_x_corr, (V_x_C - V_A) * r_p / (r_p + r_Delta) + V_A)
eq_V_x_corr = sp.Eq(V_x_corr, (V_x_C - V_A) * (r_p / (r_p + r_Delta))**2 + V_A)

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
eq_X_R = sp.Eq(X_R, -(-L_R * sin(alpha_f) + D_R * cos(alpha_f)))
eq_Y_R = sp.Eq(Y_R, (L_R * cos(alpha_f) + D_R * sin(alpha_f)))
eq_N_R = sp.Eq(N_R, x_R * Y_R)
#eq_alpha_f = Eq(alpha_f, kappa * gamma)
eq_alpha_f = Eq(alpha_f, eq_alpha.rhs.subs(delta,0))  # setting delta=0 puts the rudder and ship in the same orientation.
