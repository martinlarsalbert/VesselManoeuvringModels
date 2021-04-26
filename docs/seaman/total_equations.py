import sympy as sp
from seaman_symbols import *
import seaman_symbol as ss

import surge_hull_equations as surge_hull_equations
import sway_hull_equations as sway_hull_equations
import yaw_hull_equations as yaw_hull_equations
import roll_hull_equations as roll_hull_equations


import rudder_equations as rudder_equations

"""
Surge equation:
"""

# Todo: Seaman has some strange extra resistance due to yaw rate
#  shipdict2.design_particulars['lcg'] = 0.0 was needed to get correct drag compared to the equation below.

surge_equation = sp.Eq(lhs = X.bis,rhs = X_h.bis + n_rud*X_rudder.bis + n_prop*T_prop.bis)

eqs = (surge_equation,
       surge_hull_equations.surge_hull_equation,
       rudder_equations.rudder_drag_equation_expanded,
      )

solution = sp.solve(eqs,X.bis,X_h.bis,X_rudder.bis, simplify=False)[X.bis]
surge_equation_expanded = sp.Eq(lhs = X.bis,rhs = solution)

surge_equation_expanded_SI = ss.expand_bis(surge_equation_expanded)

"""
Sway equation:
"""

sway_equation = sp.Eq(lhs = Y.bis,rhs = (Y_h.bis + n_rud*Y_rudder.bis))

eqs = (sway_equation,
       sway_hull_equations.sway_hull_equation,
       sway_hull_equations.sway_drift_equation,
       sway_hull_equations.sway_yaw_rate_equation,
       sway_hull_equations.simplified_sway_none_linear_equation_bis,

       rudder_equations.rudder_total_sway_equation,

       )

solution = sp.solve(eqs,Y.bis,Y_h.bis,Y_rudder.bis,Y_r.bis,Y_v.bis, simplify=False)[Y.bis]
solution = solution.subs(Y_nl.bis,Y_nl.bis_eq.rhs)
Y_nl_simplified = sp.solve(sway_hull_equations.simplified_sway_none_linear_equation,Y_nl, simplify=False)[0].as_sum(20)
solution = solution.subs(Y_nl,Y_nl_simplified)

sway_equation_expanded = sp.Eq(lhs = Y.bis,rhs = solution)
sway_equation_expanded_SI = ss.expand_bis(sway_equation_expanded)

"""
Yaw equation:
"""

yaw_equation = sp.Eq(lhs = N.bis,rhs = (N_h.bis + n_rud*N_rudder.bis))

rudder_yaw_equation_bis = ss.reduce_bis(rudder_equations.rudder_yaw_equation)

eqs = (yaw_equation,

       yaw_hull_equations.yaw_hull_equation,
       yaw_hull_equations.yaw_drift_equation,
       yaw_hull_equations.yaw_yaw_rate_equation,
       yaw_hull_equations.simplified_yaw_none_linear_equation_bis,

       rudder_yaw_equation_bis,
       rudder_equations.rudder_yaw_equation_expanded,

       )

solution = sp.solve(eqs, N.bis,
                    N_h.bis, N_r.bis, N_v.bis,
                    N_rudder.bis, Y_rudder.bis, simplify=False
                    )[N.bis]

solution = solution.subs(N_nl.bis, N_nl.bis_eq.rhs)
N_nl_simplified = sp.solve(yaw_hull_equations.simplified_yaw_none_linear_equation, N_nl, simplify=False)[0].as_sum(20)
solution = solution.subs(N_nl, N_nl_simplified)

yaw_equation_expanded = sp.Eq(lhs=N.bis, rhs=solution)

yaw_equation_expanded_SI = ss.expand_bis(yaw_equation_expanded)

"""
Roll equation:
"""

roll_equation = sp.Eq(lhs = K.bis,rhs = (K_h.bis + n_rud*K_rudder.bis))

rudder_roll_equation_bis = ss.reduce_bis(rudder_equations.rudder_roll_equation)

eqs = (roll_equation,
       roll_hull_equations.roll_hull_equation,

       rudder_roll_equation_bis,
       rudder_equations.rudder_total_sway_equation,
       #rudder_equations.rudder_u_equation_effective,
       #rudder_equations.rudder_T_equation,

       )

solution = sp.solve(eqs, K.bis,
                    K_h.bis,
                    K_rudder.bis, Y_rudder.bis,
                    simplify=False, rational=False)[K.bis]

roll_equation_expanded = sp.Eq(lhs=K.bis, rhs=solution)
roll_equation_expanded_SI = ss.expand_bis(roll_equation_expanded)
