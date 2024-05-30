import sympy as sp
from sympy import Eq, symbols, Symbol, cos, sin, Derivative, atan, Piecewise, pi
from vessel_manoeuvring_models.symbols import *

eq_w_p = sp.Eq(
    (1 - w_p) / (1 - w_p0), 1 + (1 - sp.exp(-C_1 * sp.Abs(beta_p))) * (C_2 - 1)
)
eq_beta_p = Eq(beta_p, beta - x_p * r)
eq_beta = Eq(beta, sp.atan2(-v,u))

w_f = sp.symbols("w_f")
eq_w_f = Eq(w_f,sp.solve(eq_w_p,w_p)[0])

C_2_beta_p_pos,C_2_beta_p_neg = symbols("C_2_beta_p_pos,C_2_beta_p_neg")
eq_C_2 = sp.Eq(C_2,
              Piecewise(
                  (C_2_beta_p_pos,beta_p>0),
                  (C_2_beta_p_neg,beta_p<=0),
              ))