import sympy as sp
from seaman_symbols import *
import seaman_symbol as ss

cc1 = sp.sqrt(g*L)
cc2 = L*cc1

roll_hull_equation = sp.Eq(lhs = K_h.bis,
                           rhs = (
                                 (
                                    (K_ur + K_uur*u_w.bis)*u_w.bis*r_w.bis
		                          + (K_uv + K_uuv*u_w.bis)*u_w.bis*v_w.bis
		                          + K_up*u_w.bis*p.bis
		                          + K_p*p.bis
                                 )
                                +
                                  (
                                  K_vav*v_w.bis*sp.Abs(v_w.bis)
                                  + K_rar*r_w.bis*sp.Abs(r_w.bis)
                                  + K_pap*p.bis*sp.Abs(p.bis)
                                  )
                                 #ToDo: Check this:
#                                 (
#                                    (K_ur + K_uur/cc1*u_w.bis)*u_w.bis*r_w.bis
#		                          + (K_uv + K_uuv/cc2*u_w.bis)*u_w.bis*v_w.bis
#		                          + K_up*u_w.bis*p.bis
#		                          + K_p*p.bis*cc1
#                                )
#                                +
#                                  (
#                                  K_vav/L*v_w.bis*sp.Abs(v_w.bis)
#                                  + K_rar*L*r_w.bis*sp.Abs(r_w.bis)
#                                  + K_pap*L*p.bis*sp.Abs(p.bis)
#                                  )

                           )
                           )

roll_hull_equation_SI = ss.expand_bis(roll_hull_equation)