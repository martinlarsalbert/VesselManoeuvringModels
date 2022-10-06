import sympy as sp
from vessel_manoeuvring_models.symbols import *
from vessel_manoeuvring_models.parameters import df_parameters

p = df_parameters["symbol"]

## Matrices:
eq_C_A = sp.Eq(
    C_A,
    sp.UnevaluatedExpr(
        sp.matrices.MutableDenseMatrix(
            [
                [0, 0, p.Yvdot * v + p.Yrdot * r],
                [0, 0, -p.Xudot * u],
                [-p.Yvdot * v - p.Yrdot, p.Xudot * u, 0],
            ]
        )
    ),
)

eq_C_RB = sp.Eq(
    C_RB,
    sp.UnevaluatedExpr(
        sp.matrices.MutableDenseMatrix(
            [
                [0, -m * r, -m * x_G * r],
                [m * r, 0, 0],
                [m * x_G * r, 0, 0],
            ]
        )
    ),
)

eq_M_A = sp.Eq(
    M_A,
    sp.UnevaluatedExpr(
        -sp.matrices.MutableDenseMatrix(
            [
                [p.Xudot, 0, 0],
                [0, p.Yvdot, p.Yrdot],
                [0, p.Nvdot, p.Nrdot],
            ]
        )
    ),
)

eq_M_RB = sp.Eq(
    M_RB,
    sp.UnevaluatedExpr(
        sp.matrices.MutableDenseMatrix(
            [
                [m, 0, 0],
                [0, m, m * x_G],
                [0, m * x_G, I_z],
            ]
        )
    ),
)

eq_D_function = sp.Eq(
    -D_function + tau,
    sp.UnevaluatedExpr(sp.matrices.MutableDenseMatrix([[X_D, Y_D, N_D]]).transpose()),
)

eq_nu_3dof = sp.Eq(nu, sp.UnevaluatedExpr(sp.Matrix([[u, v, r]]).transpose()))
eq_nu_1d_3dof = sp.Eq(
    nu1d, sp.UnevaluatedExpr(sp.Matrix([[u1d, v1d, r1d]]).transpose())
)

A = (eq_M_A.rhs + eq_M_RB.rhs).doit()

eq_system = sp.Eq(
    sp.UnevaluatedExpr(A) * sp.UnevaluatedExpr(eq_nu_1d_3dof.rhs),
    sp.UnevaluatedExpr(
        -eq_C_RB.rhs.doit() * eq_nu_3dof.rhs.doit() + eq_D_function.rhs.doit()
    ),
)

A_inv = A.inv()
S = sp.symbols("S")
eq_S = sp.Eq(S, -sp.fraction(A_inv[1, 1])[1])

A_inv_S = A_inv.subs(eq_S.rhs, S)
eq_acceleration_matrix_clean = sp.Eq(
    sp.UnevaluatedExpr(eq_nu_1d_3dof),
    sp.UnevaluatedExpr(A_inv_S) * sp.UnevaluatedExpr(eq_system.rhs),
)

## State space model

x, x1d = sp.symbols(r"\mathbf{x} \dot{\mathbf{x}}")  # State vector
h = sp.symbols("h")
u_input = sp.symbols(r"u_{input}")  # input vector
w_noise = sp.symbols(r"w_{noise}")  # input vector

f = sp.Function("f")(x, u_input, w_noise)
eq_state_space = sp.Eq(x1d, f)
eq_x = sp.Eq(x, sp.UnevaluatedExpr(sp.Matrix([x_0, y_0, psi, u, v, r])))

eq_x0_1d = sp.Eq(x_01d, u * sp.cos(psi) - v * sp.sin(psi))
eq_y0_1d = sp.Eq(y_01d, u * sp.sin(psi) + v * sp.cos(psi))
eq_psi_1d = sp.Eq(psi1d, r)

eq_f = sp.Eq(
    f,
    sp.UnevaluatedExpr(
        sp.Matrix(
            [
                eq_x0_1d.rhs,
                eq_y0_1d.rhs,
                eq_psi_1d.rhs,
                u1d,
                v1d,
                r1d,
            ]
        )
    ),
)
