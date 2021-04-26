import sympy as sp
from seaman_symbols import *
from rudder_equations import *


rudder_total_sway_function = sp.lambdify((delta,u_w,v_w,r_w,s,T_prop, n_prop,Y_Tdelta,Y_uudelta,k_r,k_v,volume,rho,L,g,xx_rud,l_cg),
            sp.solve(rudder_total_sway_equation_SI,Y_rudder,simplify=False)[0],
                          modules='numpy',
                          )

effective_rudder_angle_function = sp.lambdify((delta, u_w, v_w, r_w, k_r, k_v, L, g, xx_rud, l_cg),
                                              sp.solve(effective_rudder_angle_equation_SI,delta_e, simplify=False)[0],
                                              modules='numpy',
                                              )

rudder_drag_function = sp.lambdify((delta, u_w, v_w, r_w, s, T_prop, n_prop, Y_Tdelta, Y_uudelta, k_r, k_v, X_Yrdelta, volume, rho, L,
                                    g, xx_rud,l_cg),
                                   sp.solve(rudder_drag_equation_expanded_SI,X_rudder, simplify=False)[0],
                                   modules='numpy',
                                   )

rudder_yawing_moment_function = sp.lambdify((delta, u_w, v_w, r_w, s, T_prop, n_prop, Y_Tdelta, Y_uudelta, k_r, k_v, volume, rho, L,
                                             g, xx_rud, l_cg),
                                            sp.solve(rudder_yaw_equation_expanded_SI,N_rudder, simplify=False)[0],
                                            modules='numpy',
                                            )

rudder_roll_moment_function = sp.lambdify((delta, u_w, v_w, r_w, s, T_prop, n_prop, Y_Tdelta, Y_uudelta, k_r, k_v, volume, rho, L, g,
                                           xx_rud, zz_rud, l_cg),
                                          sp.solve(rudder_roll_equation_expanded_SI,K_rudder, simplify=False)[0],
                                          modules='numpy',
                                          )