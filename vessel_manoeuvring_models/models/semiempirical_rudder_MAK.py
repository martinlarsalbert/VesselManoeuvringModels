"""Semi-empirical rudder model as described by Kjellber (2023).
[1] Matusiak, J., 2021. Dynamics of a Rigid Ship - with applications. Aalto University.
[2] Yasukawa, H., Yoshimura, Y., 2015. Introduction of MMG standard method for ship maneuvering predictios. J Mar Sci Technol 20, 37–52. https://doi.org/10.1007/s00773-014-0293-y
[3] Kjellberg, M., Gerhardt, F., Werner, S., n.d. Sailing Performance of Wind-Powered Cargo Vessel in Unsteady Condi- tions.
"""

import sympy as sp
from sympy import Eq, symbols, Symbol, cos, sin, Derivative, atan, Piecewise, pi
from vessel_manoeuvring_models.symbols import *
from copy import deepcopy
from vessel_manoeuvring_models.models.subsystem import EquationSubSystem
from vessel_manoeuvring_models.substitute_dynamic_symbols import lambdify, run

# ____________________________________________________________________
# Rudder model

L, D, C_L, C_D, A_R, b_R, kappa, C_L_tune = symbols(
    "L,D,C_L,C_D,A_R,b_R,kappa,C_L_tune"
)

V_R_x = symbols("V_{R_{x}}")
V_R_y = symbols("V_{R_{y}}")
V_R = symbols("V_R")
l_R = sp.symbols("l_R")  # MMG lever arm eq. 24 [2] 
V_A = symbols("V_A")
w_f, rho = sp.symbols("w_f,rho,")
x_R, y_R, z_R = sp.symbols("x_R y_R z_R")
V_x = sp.symbols("V_x")
eq_V_R = sp.Eq(V_R, sp.sqrt(V_R_x**2 + V_R_y**2))
eq_V_R_x = sp.Eq(V_R_x, V_x + q * z_R - r * y_R)
eq_V_R_y = sp.Eq(V_R_y, -v - r * l_R + p * z_R)              
eq_V_x_no_propeller = Eq(V_x,V_A)
eq_V_A = sp.Eq(V_A, (1 - w_f) * u)

# The expressions for rudder lift and drag forces are:
eq_L = Eq(L, C_L_tune * 1 / 2 * rho * C_L * A_R * V_R**2)  # eq.46
eq_D = Eq(D, 1 / 2 * rho * C_D * A_R * V_R**2)   # eq.47

C_L_no_stall = symbols("C_{L_{nostall}}")
C_D_no_stall = symbols("C_{D_{nostall}}")


alpha = symbols("alpha")  # angle of attack
C_DC = symbols("C_DC")   # crossflow drag coefficient
AR_e = symbols("AR_e")  # Effective aspect ratio

# When the angle of attack α is below the stall angle αs, the lift and drag coefficients are:
eq_CL_no_stall = Eq(C_L_no_stall,Derivative(C_L,alpha)*alpha + C_DC/AR_e*alpha*sp.Abs(alpha))  # eq.48 (|α|<αs)

C_D0 = symbols("C_D0")  # drag coefficient at zero angle of attack
e_0 = symbols("e_0")  # Oswald efficiency factor
eq_CD_no_stall = Eq(C_D_no_stall,C_D0 + C_L**2/(pi*AR_e*e_0))  # eq.49 (|α|<αs)

C_F = symbols("C_F")  # frictional coeffient at zerp angle of attack.
eq_C_D0 = Eq(C_D0, sp.Rational(2.5)*C_F)

Re_F = symbols("Re_F")  # Reynold number based on the rudder's mean chord.
eq_C_F = Eq(C_F, 0.075/(sp.log(Re_F-2,10)**2))
nu, c = sp.symbols("nu c")  # kinematic_viscosity, coord length
eq_Re_F = sp.Eq(Re_F, V_R * c / nu)
c_t = symbols("c_t")  # tip chord
c_r = symbols("c_r")  # root chord
eq_c = sp.Eq(c, (c_r + c_t)/2)

Lambda = symbols("lambda") # taper ration of the rudder
eq_Lambda = Eq(Lambda, c_t/c_r)  # taper ration of the rudder
eq_C_DC = Eq(C_DC, 0.1 + 1.6*Lambda)  # eq.51 square tip

delta, delta_lim = symbols("delta delta_lim")
AR_g = symbols("AR_g")  # Geometric aspect ratio?
# Assuming the gap at the rudder root is small at rudder angle δ = 0 and large at maximum rudder angle δmax, 
# the effective aspect ratio of the rudder is taken as:
eq_AR_e = Eq(AR_e, (2-delta/delta_lim)*AR_g)  # eq.52
eq_AR_g = Eq(AR_g, b_R**2 / A_R)

# The lift curve slope of the rudder is given by:
a_0 = symbols("a_0")  # section lift curve slope
Omega = symbols("Omega")  # sweep angle of the quarter chord line (see Lewis, 1989)
eq_C_L_alpha = Eq(Derivative(C_L,alpha), a_0*AR_e/(1.8 + cos(Omega)*sp.sqrt(AR_e**2/(cos(Omega)**4)+4)))

eq_a0 = Eq(a_0, 0.9*2*pi) # section lift curve slope

# Following Lloyd (1989), the stall angle is given by:
alpha_s = symbols("alpha_s")

#eq_alpha_s = Eq(alpha_s, sp.Rational(1.225) - sp.Rational(0.445)*AR_e + sp.Rational(0.075)*AR_e**2)  #AR_e < 3.0
#eq_alpha_s = Eq(alpha_s, sp.Rational(0.565))  #AR_e > 3.0
eq_alpha_s = Eq(alpha_s, Piecewise(
    (1.225 - 0.445*AR_e + 0.075*AR_e**2, AR_e <= 3),
    (0.565, AR_e > 3)
))
C_N = symbols("C_N")  # normal force coefficient for a stalling wing
eq_C_N = Eq(C_N, sp.Rational(1.8))  # recommended by Hoerner and Borst (1975)
B_s,B_0 = symbols("B_s,B_0")
C_L_max = sp.Symbol("C_{L_{max}}")
C_D_max = sp.Symbol("C_{D_{max}}")
C_L_stall = symbols("C_{L_{stall}}")
C_D_stall = symbols("C_{D_{stall}}")

eq_CL_stall = Eq(C_L_stall, C_N*sin(alpha)*cos(alpha)*B_s + C_L_max*B_0) # eq.55 (|α|>=αs)
eq_CD_stall = Eq(C_D_stall, C_N*sin(alpha)**2*B_s + C_D_max*B_0) # eq.56 (|α|>=αs)

u_s = symbols("u_s")
eq_B_s = Eq(B_s, Piecewise(
    (1,sp.Abs(alpha) > sp.Rational(1.25)*alpha_s),
    ((3*u_s-2*u_s**2)*u_s, sp.Abs(alpha) <= 1.25*alpha_s)
))
eq_u_s = Eq(u_s,4*(sp.Abs(alpha)-alpha_s)/alpha_s)

eq_B_0 = Eq(B_0,1-B_s)
eq_CL_max = Eq(C_L_max,Derivative(C_L,alpha)*alpha_s + C_DC/AR_e*alpha_s*sp.Abs(alpha_s))  # eq.58
eq_CD_max = Eq(C_D_max, C_D0 + C_L_max**2/(pi*AR_e*e_0))

lambda_R = symbols("lambda_R")  # Limited radius of slipstream (if any)
eq_C_L = Eq(C_L, lambda_R*Piecewise(
    (eq_CL_no_stall.rhs, sp.Abs(alpha)<alpha_s),
    (eq_CL_stall.rhs, sp.Abs(alpha) >= alpha_s)
))

eq_C_D = Eq(C_D, Piecewise(
    (eq_CD_no_stall.rhs, sp.Abs(alpha) < alpha_s),
    (eq_CD_stall.rhs, sp.Abs(alpha) >= alpha_s)
))

gamma=symbols("gamma")
kappa=symbols("kappa")
#eq_alpha = Eq(alpha, -delta+kappa*gamma)  # why -delta?
eq_alpha = Eq(alpha, delta+kappa*gamma)
V_y = symbols("V_y")
eq_gamma = Eq(gamma, atan(V_R_y/V_R_x))

# ____________________________________________________________________
# The effect of propeller action on the rudder flow
V_x_corr = symbols("V_{x_{corr}}")
V_infty, C_Th, r_0 = sp.symbols("V_\infty,C_Th,r_0")
eq_V_infty = sp.Eq(V_infty, V_A * sp.sqrt(1 + C_Th))
eq_C_Th = sp.Eq(
    C_Th,
    thrust / n_prop / (sp.Rational(1, 2) * rho * V_A**2 * pi * (2 * r_0) ** 2 / 4),
)
r_infty = sp.symbols("r_\infty")
eq_r_infty = sp.Eq(r_infty, r_0 * sp.sqrt(sp.Rational(1, 2) * (1 + V_A / V_infty)))
r_p, x = sp.symbols("r_p,x")
r_x = r_p  # (r_x MAK, r_p matusiak)
eq_r = sp.Eq(
    r_p,
    r_0
    * (0.14 * (r_infty / r_0) ** 3 + (r_infty / r_0) * (x / r_0) ** 1.5)
    / ((0.14 * r_infty / r_0) ** 3 + (x / r_0) ** 1.5),
)
eq_V_x = sp.Eq(V_x, V_infty * (r_infty / r_p) ** 2)
r_Delta = sp.symbols("r_Delta")
Delta_r_x = r_Delta  #(MAK, Matusiak)
eq_r_Delta = sp.Eq(r_Delta, 0.15 * x * ((V_x - V_A) / (V_x + V_A)))
eq_V_x_corr = sp.Eq(V_x_corr, (V_x - V_A) * r_p / (r_p + r_Delta) + V_A)

#The limited radius of the slipstream in the lateral direction also diminishes the 
# rudder lift force, which can be taken into account by multiplying the lift coefficient 
# CL by a factor λ, given by:
f = symbols("f")
d = symbols("d")
c = symbols("c")  # mean chord
eq_lambda_R = Eq(lambda_R,(V_A/V_x_corr)**f)  #eq.67
eq_f = Eq(f, 2*(2/(2+d/c))**8)
eq_d = Eq(d, sp.sqrt(pi/4)*(r_x+Delta_r_x))
