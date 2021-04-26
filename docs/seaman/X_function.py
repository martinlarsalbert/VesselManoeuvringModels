from numpy import *
def X_function(delta, u_w, v_w, r_w, s, T_prop, n_prop, X_res, X_vv, X_rr, X_vr, X_Yrdelta, Y_Tdelta, Y_uudelta, k_r, k_v, Y_uv, Y_uuv, Y_ur, Y_uur, C_d, t_a, t_f, disp, rho, L, g, xx_rud, l_cg, n_rud):
    return (7.28*T_prop*delta**2*u_w*X_Yrdelta*Y_Tdelta*n_rud*sqrt(L*g)/(L*g) + T_prop*delta**2*X_Yrdelta*Y_Tdelta*n_rud + T_prop*n_prop + X_res + delta**2*u_w**2*X_Yrdelta*Y_uudelta*n_rud*disp*rho/L + r_w**2*X_rr*L*disp*rho + r_w*v_w*X_vr*disp*rho*sqrt(L*g)/(L*sqrt(g/L)) + v_w**2*X_vv*disp*rho/L)
