from typing import Any
import sympy as sp
from sympy import Eq, symbols, Symbol, cos, sin, Derivative, atan, Piecewise, pi
from vessel_manoeuvring_models.symbols import *

eq_X_P = sp.Eq(X_P, (1 - tdf) * thrust)
eq_Y_P = sp.Eq(Y_P, 0)
eq_N_p = sp.Eq(N_P, 0)

eq_T = sp.Eq(thrust, rho * rev ** 2 * D ** 4 * K_T)
eq_Q = sp.Eq(torque, rho * rev ** 2 * D ** 5 * K_Q)
eq_K_T = sp.Eq(K_T, k_2 * J ** 2 + k_1 * J + k_0)
eq_K_Q = sp.Eq(K_Q, k_q2 * J ** 2 + k_q1 * J + k_q0)
eq_J = sp.Eq(J, u * (1 - w_f) / (rev * D))