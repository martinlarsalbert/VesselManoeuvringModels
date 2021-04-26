import sympy as sp
from seaman_symbols import *
import seaman_symbol as ss

"""
Y force
"""
rudder_equation_no_stall = sp.Eq(lhs = Y_rudder.bis,rhs = (Y_rudder_u.bis + Y_rudder_T.bis))

rudder_equation = sp.Eq(lhs = Y_rudder.bis,rhs = (Y_rudder_u.bis + Y_rudder_T.bis)*(1+s*delta_e**2))

effective_rudder_angle_equation = sp.Eq(lhs=delta_e,
                                        rhs=delta + k_v*(v_w.bis/u_w.bis) - k_r*((xx_rud.bis - l_cg.bis)*r_w.bis/u_w.bis))

# Seaman Manual version:
#effective_rudder_angle_equation = sp.Eq(lhs=delta_e,
#                                        rhs=delta + (k_v*(v_w.bis/u_w.bis) + k_r*((xx_rud.bis - l_cg.bis)*r_w.bis/u_w.bis))*sp.Abs(delta))

delta_e_expanded = sp.solve(effective_rudder_angle_equation,delta_e)[0]
effective_rudder_angle_equation_SI = ss.expand_bis(effective_rudder_angle_equation)


rudder_u_equation = sp.Eq(lhs = Y_rudder_u.bis,rhs = Y_uudelta*u_w.bis**2*delta_e)

# ToDo: Check this:
rudder_T_equation = sp.Eq(lhs = Y_rudder_T.bis,rhs = Y_Tdelta*T_prop.bis*delta*(1. + 1.82*u_w.bis / 0.25))  # The (1. + 1.82*u_w.bis / 0.25)) is most likely wrong in seaman

# Solve:
eqs = [
    rudder_equation.subs(delta_e,delta_e_expanded),
    rudder_u_equation.subs(delta_e,delta_e_expanded),
    rudder_T_equation.subs(delta_e,delta_e_expanded),
]
solution = sp.solve(eqs,(Y_rudder.bis,Y_rudder_u.bis,Y_rudder_T.bis,))[Y_rudder.bis]
rudder_total_sway_equation = sp.Eq(lhs = Y_rudder.bis,rhs = solution)
rudder_total_sway_equation_SI = ss.expand_bis(rudder_total_sway_equation)


"""
X Force
"""

rudder_drag_equation = sp.Eq(lhs = X_rudder.bis,rhs = (X_Yrdelta*((Y_rudder.bis))*delta))

eqs = [
    rudder_drag_equation.subs(delta_e,delta_e_expanded),
    rudder_equation_no_stall.subs(delta_e,delta_e_expanded),  # No stall!!
    rudder_u_equation.subs(delta_e,delta_e_expanded),
    rudder_T_equation.subs(delta_e,delta_e_expanded),
]
solution = sp.solve(eqs,(X_rudder.bis,Y_rudder.bis,Y_rudder_u.bis,Y_rudder_T.bis,),simplify=False)[X_rudder.bis]
rudder_drag_equation_expanded = sp.Eq(lhs = X_rudder.bis,rhs = solution)
rudder_drag_equation_expanded_old = rudder_drag_equation_expanded.copy()
rudder_drag_equation_expanded = rudder_drag_equation_expanded.subs([(k_v,0),
                                     (k_r,0)                       
                                        ])
rudder_drag_equation_expanded_SI = ss.expand_bis(rudder_drag_equation_expanded)
rudder_drag_equation_expanded_old_SI = ss.expand_bis(rudder_drag_equation_expanded_old)

"""
N Moment
"""
rudder_yaw_equation = sp.Eq(lhs = N_rudder.bis, rhs =Y_rudder.bis * (xx_rud.bis))
eqs = [

    rudder_total_sway_equation,
    rudder_yaw_equation

]

solution = sp.solve(eqs, (N_rudder.bis, Y_rudder.bis))[N_rudder.bis]
rudder_yaw_equation_expanded = sp.Eq(lhs=N_rudder.bis, rhs=solution)
rudder_yaw_equation_expanded_SI = ss.expand_bis(rudder_yaw_equation_expanded)

"""
K Moment
"""
rudder_roll_equation = sp.Eq(lhs = K_rudder.bis, rhs =-Y_rudder.bis*(zz_rud.bis+t_a.bis))
eqs = [

    rudder_total_sway_equation,
    rudder_roll_equation,

]

solution = sp.solve(eqs, (K_rudder.bis, Y_rudder.bis))[K_rudder.bis]
rudder_roll_equation_expanded = sp.Eq(lhs=K_rudder.bis, rhs=solution)
rudder_roll_equation_expanded_SI = ss.expand_bis(rudder_roll_equation_expanded)