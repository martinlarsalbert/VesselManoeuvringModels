import sympy as sp
from src.symbols import *

eq_T = sp.Eq(thrust, rho * rev ** 2 * D ** 4 * K_T)
eq_K_T = sp.Eq(K_T, k_2 * J ** 2 + k_1 * J + k_0)
eq_J = sp.Eq(J, u * (1 - w_p) / (rev * D))

eqs = [
    eq_T,
    eq_J,
    eq_K_T,
]

# Thrust
solution = sp.solve(eqs, thrust, K_T, J, dict=True)[0][thrust]
eq_thrust_simple = sp.Eq(thrust, solution)
lambda_thrust_simple = sp.lambdify(
    list(eq_thrust_simple.rhs.free_symbols), eq_thrust_simple.rhs
)

# w_p
solution = sp.solve(eq_thrust_simple, w_p, dict=True)[1][w_p]
eq_w_p = sp.Eq(w_p, solution)
lambda_w_p = sp.lambdify(list(eq_w_p.rhs.free_symbols), eq_w_p.rhs)
