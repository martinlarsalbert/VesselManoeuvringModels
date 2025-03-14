"""
[1] Fossen, T.I., 2011. Handbook of Marine Craft Hydrodynamics and Motion Control. John Wiley & Sons.

Also see: 155.01_Fossen_model.ipynb

"""

import sympy as sp
from sympy import Eq,symbols
from vessel_manoeuvring_models.symbols import *
from sympy import ImmutableDenseMatrix
from vessel_manoeuvring_models.parameters import df_parameters
p = df_parameters["symbol"]

M,C,D,tau,tau_wind,tau_wave,upsilon,upsilon1d = symbols(r"\mathbf{M},\mathbf{C},\mathbf{D},\mathbf{\tau},\mathbf{\tau_{wind}},\mathbf{\tau_{wave}},\mathbf{\upsilon},\dot{\mathbf{\upsilon}}")
M_RB, M_A = symbols(r"\mathbf{M_{RB}},\mathbf{M_A}")
C_RB, C_A = symbols(r"\mathbf{C_{RB}},\mathbf{C_A}")

eq_main = Eq(M*upsilon1d + C*upsilon ,D + tau + tau_wind + tau_wave)  # (6.4)
eq_M = Eq(M, M_RB+M_A)
eq_C = Eq(C, C_RB+C_A)

eq_M_RB = Eq(M_RB,ImmutableDenseMatrix([
    
    [m,0    ,0],
    [0,m    ,m*x_G],
    [0,m*x_G,I_z],
    
]), evaluate=False)  # (6.7)

eq_C_RB = Eq(C_RB,ImmutableDenseMatrix([
    
    [0,-m*r,-m*x_G*r],
    [m*r,0,0],
    [m*x_G*r,0,0],
    
]), evaluate=False)

eq_M_A = Eq(M_A,-ImmutableDenseMatrix([
    
    [p.Xudot,0,0],
    [0,p.Yvdot,p.Yrdot],
    [0,p.Nvdot,p.Nrdot],
    
]), evaluate=False)  # (6.50)

eq_C_A = Eq(C_A,ImmutableDenseMatrix([
    
    [0,0,p.Yvdot*v+p.Yrdot*r],
    [0,0,-p.Xudot*u],
    [-p.Yvdot*v-p.Yrdot*r,p.Xudot*u,0],
    
]), evaluate=False)  # (6.51)

eq_upsilon = Eq(upsilon,ImmutableDenseMatrix([u,v,r]), evaluate=False)  # velocity vector
eq_upsilon1d = Eq(upsilon1d,ImmutableDenseMatrix([u1d,v1d,r1d]), evaluate=False)  # acceleration vector

def subs_lhs_rhs(eq_main, equations):
    for eq in equations:
        eq_main = eq_main.subs(eq.lhs,eq.rhs)

    return eq_main

eq_main_expanded = subs_lhs_rhs(eq_main, [eq_M,eq_C])

eq_D = Eq(D,sp.ImmutableDenseMatrix([X_D,Y_D,N_D]), evaluate=False)

eq_main_expanded_matrix = Eq(
    (eq_M_A.rhs + eq_M_RB.rhs)*eq_upsilon1d.rhs + (eq_C_A.rhs + eq_C_RB.rhs)*eq_upsilon.rhs,
    eq_D.rhs, evaluate=False)