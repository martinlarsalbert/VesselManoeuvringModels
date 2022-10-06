"""[summary]

[1] Clarke, D., P. Gedling, and G. Hine. “The Application of Manoeuvring Criteria in Hull Design Using Linear Theory.” Transactions of the Royal Institution of Naval Architects, RINA., 1982. https://repository.tudelft.nl/islandora/object/uuid%3A2a5671ac-a502-43e1-9683-f27c50de3570.
[2] Brix, Jochim E. Manoeuvring Technical Manual. Seehafen-Verlag, 1993.

"""

import sympy as sp
import sympy.physics.mechanics
from vessel_manoeuvring_models.substitute_dynamic_symbols import lambdify
import pandas as pd
import numpy as np
from sympy import MatrixSymbol

# def sp.symbols(s):
#    """Overlaying this to append a fix for pickle"""
#
#    symbols = sympy.physics.mechanics.sp.symbols(s)
#
#    try:
#        iter(symbols)
#    except TypeError:
#        symbols.__class__.__module__ = "__main__"
#    else:
#        for symbol in symbols:
#            symbol.__class__.__module__ = "__main__"
#
#    return symbols


"""Fossen, T., 2011. Nonlinear maneuvering theory and path-following control. Marine Technology and Engineering 1, 445–460.
"""
M = sp.symbols("M")  # Inertia matrix

C = sp.symbols("C")  # Coriolis matrix
D = sp.symbols("D")  # Damping matrix


x0_, y0, z0, phi, theta, psi = sp.symbols("x0, y0, z0, phi, theta, psi")
u, v, w, p, q, r = sp.symbols("u, v, w, p, q, r")
u_c, v_c, w_c = sp.symbols("u_c, v_c, w_c")

eta = sp.symbols("eta")  # Positions
nu = sp.symbols("nu")  # Velocities
nu1d = sp.symbols(r"\dot{\nu}")

nu_r = sp.symbols("nu_r")  # Relative velocities (including current)
nu_r1d = sp.symbols(r"\dot{\nu_r}")  # Relative velocities (including current)

nu_c = sp.symbols("nu_c")  # Velocities (current)
nu_c1d = sp.symbols(r"\dot{\nu_c}")  # Velocities (current)


tau, tau_wind, tau_wave = sp.symbols("tau, tau_wind, tau_wave")
g = sp.symbols("g")  # Gravitational and buoyancy forces.
g_0 = sp.symbols("g_0")  # Static restoring forces and moments due to ballast systems

C_function = sp.Function("C")(nu)  # Coriolis matrix
D_function = sp.Function("D")(nu)  # Damping matrix
g_function = sp.Function("g")(eta)  # Damping matrix


eq_6DOF = sp.Eq(
    M * nu1d + C_function * nu + D_function * nu + g_function + g_0,
    tau + tau_wind + tau_wave,
)
#
eq_eta = sp.Eq(
    eta, sp.UnevaluatedExpr(sp.Matrix([[x0_, y0, z0, phi, theta, psi]]).transpose())
)
eq_nu = sp.Eq(nu, sp.UnevaluatedExpr(sp.Matrix([[u, v, w, p, q, r]]).transpose()))

eq_nu_c = sp.Eq(
    nu_c, sp.UnevaluatedExpr(sp.Matrix([[u_c, v_c, w_c, 0, 0, 0]]).transpose())
)
eq_nu_r = sp.Eq(nu_r, nu - nu_c)
eq_nu_r_expanded = sp.Eq(nu_r, sp.UnevaluatedExpr((eq_nu.rhs - eq_nu_c.rhs).doit()))


M_RB = sp.symbols("M_RB")  # Rigid body intertia matrix
M_A = sp.symbols("M_A")  # Added mass matrix

C_RB = sp.symbols("C_RB")  # Coriolis centrepetal matrix
C_A = sp.symbols("C_A")  # Coriolis centrepetal added mass matrix

eq_M = sp.Eq(M, M_RB + M_A)
eq_C = sp.Eq(C, C_RB + C_A)

eq_6DOF_expanded = sp.Eq(
    M_RB * nu1d + M_A * nu1d + C_RB * nu + C_A * nu_r + D * nu_r + g_function + g_0,
    tau + tau_wind + tau_wave,
)

eq_nu_steady = sp.Eq(nu_r1d, nu1d)


x_0, x_01d = sp.symbols("x_0 \dot{x_0}")
y_0, y_01d = sp.symbols("y_0 \dot{y_0}")
psi, psi1d = sp.symbols("\Psi \dot{\Psi}")

u, v, r, delta, thrust = sp.symbols("u v r delta thrust")
(
    u1d,
    v1d,
    r1d,
) = sp.symbols(r"\dot{u} \dot{v} \dot{r}")


m, x_G, U, I_z, volume = sp.symbols("m x_G U I_z volume")
π = sp.pi
T, L, CB, B, rho, t, dt = sp.symbols("T L CB B rho t dt")

f_ext_x, f_ext_y, m_ext_z = sp.symbols("f_ext_x,f_ext_y,m_ext_z")  # external forces


X_X, X_Y, X_N = sp.symbols("X_X X_Y X_N")  # State matrixes

X_force, Y_force, N_force = sp.symbols("X_force Y_force N_force")  # Force models

X_D = sp.Function("X_D")(u, v, r, delta, thrust)  # damping
Y_D = sp.Function("Y_D")(u, v, r, delta, thrust)  # damping
N_D = sp.Function("N_D")(u, v, r, delta, thrust)  # damping
for item in [X_D, Y_D, N_D]:
    item.__class__.__module__ = "__main__"  # Fix for pickle

n, delta_t = sp.symbols("n delta_t")  # Time step n

n = sp.symbols("n")  # Number of points
N = sp.symbols("N")  # Numver of degrees of freedome
X_X = MatrixSymbol("X_X", n, N)
X_Y = MatrixSymbol("X_Y", n, N)
X_N = MatrixSymbol("X_N", n, N)
A_coeff = MatrixSymbol("A_coeff", N, 1)
B_coeff = MatrixSymbol("B_coeff", N, 1)
C_coeff = MatrixSymbol("C_coeff", N, 1)


X_n = sp.Function("X")(n)  # X features
Y_n = sp.Function("Y")(n)  # X features
N_n = sp.Function("N")(n)  # X features


## MMG model:
x_r = sp.symbols("x_r")  # X pos of rudder
x_p = sp.symbols("x_r")  # X pos of propeller
R0 = sp.symbols("R0")
X_H = sp.symbols("X_H")  # Hull surge force
Y_H = sp.symbols("Y_H")  # Hull sway force
N_H = sp.symbols("N_H")  # Hull yaw moment
X_R = sp.symbols("X_R")  # Rudder surge force
Y_R = sp.symbols("Y_R")  # Rudder sway force
N_R = sp.symbols("N_R")  # Rudder yaw moment
X_P = sp.symbols("X_P")  # Propeller surge force
tdf = sp.symbols("tdf")  # Thrust deduction factor
rev = sp.symbols("rev")  # propeller speed [1/s]
K_T = sp.symbols("K_T")
k_2, k_1, k_0 = sp.symbols("k_2, k_1, k_0")  # K_T coefficients.
J = sp.symbols("J")  # Propeller advance ratio
w_p = sp.symbols("w_p")  # Wake at propeller (including drift angle etc.)
w_p0 = sp.symbols("w_p0")  # Taylor wake at straight course.
C_1, C_2 = sp.symbols("C_1, C_2")
beta_p = sp.symbols(
    "beta_p"
)  # geometrical inflow angle to the propeller in maneuvering motions
F_N = sp.symbols("F_N")  # Rudder normal force
U_R, u_R, v_R = sp.symbols("U_R, u_R, v_R")  # Velocities at rudder
alpha_R = sp.symbols("alpha_R")  # inflow angle?
A_R = sp.symbols("A_R")  # Rudder aspect ratio
f_alpha = sp.symbols("f_alpha")
gamma_R = sp.symbols("gamma_R")  # flow straightening coefficient
beta_R = sp.symbols("beta_R")  # rudder geometrical inflow angle
beta = sp.symbols("beta")  # drift angle
l_R = sp.symbols("l_R")  # experimental constant for expressing vR accurately
epsilon = sp.symbols("epsilon")
kappa = sp.symbols("kappa")
t_R = sp.symbols("t_R")  # steering resistance deduction factor
a_H, x_H = sp.symbols(
    "a_H x_H"
)  # coefficients representing mainly hydrodynamic interaction between ship hull and rudder
Lambda = sp.symbols("Lambda")  # Rudder aspect ratio
H_R, C_R = sp.symbols("H_R C_R")  # Rudder height and choord.
eta = sp.symbols("eta")  # Ratio of propeller diameter to rudder span (D/H_R)
C_Th = sp.symbols("C_Th")


def glue_equations(module):
    # Glue equations
    from myst_nb import glue

    from sympy.physics.vector.printing import vlatex
    from IPython.display import Math

    for key, value in module.__dict__.items():
        if isinstance(value, sp.Eq):
            glue(key, Math(vlatex(value)), display=False)
            print(f"gluing:{key}")
