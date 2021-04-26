import sympy as sp
from seaman_symbols import *
from sway_hull_equations import *

Yv_function = sp.lambdify((u_w,v_w,Y_uv,Y_uuv,volume,rho,L,g),
            sp.solve(sway_drift_equation_SI,Y_v),)

Yr_function = sp.lambdify((u_w,r_w,Y_ur,Y_uur,volume,rho,L,g),
            sp.solve(sway_yaw_rate_equation_SI,Y_r))

Y_nl_function = sp.lambdify((rho,v_w,r_w,t_a,t_f,L,C_d),
                sp.Integral(f.subs(T_x,section_draught_equation.rhs),(x_s,-L/2,L/2)).as_sum(n = 20),
                modules = 'numpy')

Y_h_function = sp.lambdify((u_w,v_w,r_w,Y_uv,Y_uuv,Y_ur,Y_uur,t_a,t_f,rho,L,C_d,g,volume),
            total_sway_hull_equation_SI.rhs,
            modules = 'numpy',
                          )
