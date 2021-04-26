import sympy as sp
from seaman_symbols import *
from yaw_hull_equations import *
from total_lambda_functions import FunctionProvider

#Nv_function = sp.lambdify((u_w,v_w,N_uv,N_uuv,volume,rho,L,g),
#            sp.solve(yaw_drift_equation_SI,N_v),)
#
#Nr_function = sp.lambdify((u_w,r_w,N_ur,N_uur,volume,rho,L,g),
#            sp.solve(yaw_yaw_rate_equation_SI,N_r))
#
#N_nl_function = sp.lambdify((rho,v_w,r_w,t_a,t_f,L,C_d),
#                sp.Integral(f.subs(T_x,section_draught_equation.rhs),(x_s,-L/2,L/2)).as_sum(n = 20),
#                modules = 'numpy')

class HullYawFunction(FunctionProvider):

    @property
    def function(self):
        import total_equations as total_equations
        return sp.lambdify((u_w,v_w,r_w,N_uv,N_uuv,N_ur,N_uur,t_a,t_f,rho,L,C_d,g,volume,Cd_lever),
            total_yaw_hull_equation_SI.rhs,
            modules = 'numpy',
                          )

N_h_function = HullYawFunction(name = 'N_h_function').get()