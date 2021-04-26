import sympy as sp
from seaman_symbols import *
from roll_hull_equations import *

K_h_function = sp.lambdify((u_w,v_w,r_w,p,K_ur,K_uur,K_uv,K_uuv,K_up,K_p,K_vav,K_rar,K_pap,volume,rho,L,g),
            sp.solve(roll_hull_equation_SI,K_h)[0],)

