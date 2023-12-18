from . semiempirical_covered import *

## Removing the piecewise expression (they don't work in jacobians)

eq_alpha_s = Eq(
    alpha_s,
    delta_alpha_s
    #+ Piecewise(
    #    (1.225 - 0.445 * AR_e + 0.075 * AR_e**2, AR_e <= 3), (0.565, AR_e > 3)
    #),
    + 1.225 - 0.445 * AR_e + 0.075 * AR_e**2    
)

eq_B_s = Eq(
    B_s,
    #Piecewise(
    #    (1, sp.Abs(alpha) > 1.25 * alpha_s),
    #    ((3 * u_s - 2 * u_s**2) * u_s, sp.Abs(alpha) <= 1.25 * alpha_s),
    #),
    (3 * u_s - 2 * u_s**2) * u_s
)

eq_C_L = Eq(
    C_L,
    #Piecewise(
    #    (eq_CL_no_stall.rhs, sp.Abs(alpha) < alpha_s),
    #    (eq_CL_stall.rhs, sp.Abs(alpha) >= alpha_s),
    #),
    eq_CL_no_stall.rhs
)

eq_C_D_C = Eq(
    C_D_C,
    #Piecewise(
    #    (eq_CD_no_stall_C.rhs, sp.Abs(alpha) < alpha_s),
    #    (eq_CD_no_stall_C.rhs, sp.Abs(alpha) >= alpha_s),
    #),
    eq_CD_no_stall_C.rhs
)

eq_C_D_U = Eq(
    C_D_U,
    #Piecewise(
    #    (eq_CD_no_stall_U.rhs, sp.Abs(alpha) < alpha_s),
    #    (eq_CD_no_stall_U.rhs, sp.Abs(alpha) >= alpha_s),
    #),
    eq_CD_no_stall_U.rhs
)

eq_kappa = Eq(
    kappa,
    #Piecewise(
    #    (kappa_outer, ((gamma >= 0) & (y_R <= 0)) | ((gamma < 0) & (y_R > 0))),
    #    (kappa_inner, ((gamma >= 0) & (y_R > 0)) | ((gamma < 0) & (y_R <= 0))),
    #),
    (kappa_outer + kappa_inner)/2
)