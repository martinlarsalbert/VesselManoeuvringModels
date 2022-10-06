import sympy as sp
from vessel_manoeuvring_models.symbols import *

eq_Xp = sp.Eq(X_P, (1 - tdf) * thrust)
eq_T = sp.Eq(thrust, rho * rev ** 2 * D ** 4 * K_T)
eq_K_T = sp.Eq(K_T, k_2 * J ** 2 + k_1 * J + k_0)
eq_J = sp.Eq(J, u * (1 - w_p) / (rev * D))
eq_w_p = sp.Eq(
    (1 - w_p) / (1 - w_p0), 1 + (1 - sp.exp(-C_1 * sp.Abs(beta_p))) * (C_2 - 1)
)
eq_beta_p = sp.Eq(beta_p, beta - x_p * r)

eq_C_Th2 = sp.Eq(C_Th, 8 * K_T / (sp.pi * J ** 2))
eqs = [eq_T, eq_J, eq_C_Th2]
eq_C_Th1 = sp.Eq(C_Th, sp.solve(eqs, K_T, rev, C_Th, dict=True)[0][C_Th])


eqs = [
    eq_Xp,
    eq_T,
    eq_K_T,
    eq_J,
    eq_w_p.subs(beta_p, eq_beta_p.rhs),
]
X_P_solution = sp.solve(eqs, beta_p, w_p, J, K_T, thrust, X_P, dict=True)
