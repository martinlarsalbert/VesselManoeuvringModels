import sympy as sp
from seaman_symbols import *
from surge_hull_equations import *

X_h_function = sp.lambdify((v_w,r_w,X_vv,X_vr,X_rr,X_res,volume,rho,L,g),
                           sp.solve(surge_hull_equation_SI,X_h)[0],
                           modules='numpy',
                           )