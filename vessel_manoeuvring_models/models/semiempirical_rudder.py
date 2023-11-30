"""solution_lift
Semi-empirical rudder model as described by Matusiak (2021).
[1] Matusiak, J., 2021. Dynamics of a Rigid Ship - with applications. Aalto University.
[2] Yasukawa, H., Yoshimura, Y., 2015. Introduction of MMG standard method for ship maneuvering predictions. J Mar Sci Technol 20, 37–52. https://doi.org/10.1007/s00773-014-0293-y
"""
import sympy as sp
from vessel_manoeuvring_models.symbols import *
from copy import deepcopy
from vessel_manoeuvring_models.models.subsystem import EquationSubSystem
from vessel_manoeuvring_models.substitute_dynamic_symbols import lambdify, run

# ____________________________________________________________________
# Rudder model
u, v, w, p, q, r = sp.symbols("u v w p q r")
V_xR, V_yR, V_zR = sp.symbols("V_xr V_yr V_zr")
V_xWave, V_yWave, V_zWave = sp.symbols("V_xWave V_yWave V_zWave")
x_R, y_R, z_R = sp.symbols("x_R y_R z_R")
l_R = sp.symbols("l_R")  # MMG lever arm eq. 24 [2]
gamma = sp.symbols("gamma")
V_x = sp.symbols("V_x")

eq_V_xR_wave = sp.Eq(V_xR, V_x - V_xWave + q * z_R - r * y_R)
# eq_V_yR_wave = sp.Eq(V_yR, -v + V_yWave - r * x_R + p * z_R)
eq_V_yR_wave = sp.Eq(V_yR, -v + V_yWave - r * l_R + p * z_R)

eq_V_zR_wave = sp.Eq(V_zR, -w + V_zWave - q * y_R - q * x_R)
eq_V_xR = eq_V_xR_wave.subs(V_xWave, 0)
eq_V_yR = eq_V_yR_wave.subs(V_yWave, 0)
eq_V_zR = eq_V_zR_wave.subs(V_zWave, 0)

eq_gamma = sp.Eq(gamma, sp.atan(V_yR / V_xR))

V_R = sp.symbols("V_R")
eq_V_R = sp.Eq(V_R, sp.sqrt(V_xR**2 + V_yR**2 + V_zR**2))

Lambda, Lambda_g = sp.symbols("Lambda Lambda_g")
delta, delta_lim = sp.symbols("delta delta_lim")
L, D, C_L, C_D, A_R, b_R, kappa, C_L_tune = sp.symbols(
    "L,D,C_L,C_D,A_R,b_R,kappa,C_L_tune"
)

eq_Lambda_g = sp.Eq(Lambda_g, b_R**2 / A_R)
eq_Lambda = sp.Eq(Lambda, Lambda_g * (2 - sp.Abs(delta / delta_lim)))

eq_L = sp.Eq(L, C_L_tune * 1 / 2 * rho * C_L * A_R * V_R**2)
eq_D = sp.Eq(D, 1 / 2 * rho * C_D * A_R * V_R**2)

eq_C_L = sp.Eq(
    C_L,
    2
    * sp.pi
    * Lambda
    * (Lambda + 1)
    / (Lambda + 2) ** 2
    * sp.sin(delta + kappa * gamma),
)

## Drag:
C_D0 = sp.symbols("C_D0")
eq_C_D = sp.Eq(C_D, 1.1 * C_L**2 / (sp.pi * Lambda) + C_D0)

C_F, R_e = sp.symbols("C_F,R_e")
eq_C_D0 = sp.Eq(C_D0, 2.5 * C_F)

eq_CF = sp.Eq(C_F, 0.075 / ((sp.log(R_e) - 2) ** 2))

nu, c = sp.symbols("nu c")  # kinematic_viscosity, coord length
eq_Re = sp.Eq(R_e, V_R * c / nu)
eq_c = sp.Eq(c, A_R / b_R)

# ____________________________________________________________________
# The effect of propeller action on the rudder flow
V_inf, V_A, C_Th, r_0, u, w_f, rho = sp.symbols("V_inf,V_A,C_Th,r_0,u,w_f,rho,")
eq_V_A = sp.Eq(V_A, (1 - w_f) * u)
eq_V_inf = sp.Eq(V_inf, V_A * sp.sqrt(1 + C_Th))
eq_C_Th = sp.Eq(
    C_Th,
    thrust / n_prop / (sp.Rational(1, 2) * rho * V_A**2 * sp.pi * (2 * r_0) ** 2 / 4),
)
r_inf = sp.symbols("r_inf")
eq_r_inf = sp.Eq(r_inf, r_0 * sp.sqrt(sp.Rational(1, 2) * (1 + V_A / V_inf)))
r_p, x = sp.symbols("r_p,x")
# eq_r = sp.Eq(
#    r_p,
#    r_0
#    * (0.14 * (r_inf / r_0) ** 3 + (r_inf / r_0) * (x / r_0) ** 1.5)
#    / ((0.14 * r_inf / r_0) ** 3 + (x / r_0) ** 1.5),
# )
eq_r = sp.Eq(
    r_p,
    r_0
    * (0.14 * (r_inf / r_0) ** 3 + (r_inf / r_0) * (x / r_0) ** 1.5)
    / (0.14 * (r_inf / r_0) ** 3 + (x / r_0) ** 1.5),
)
eq_V_x = sp.Eq(V_x, V_inf * (r_inf / r_p) ** 2)
r_Delta = sp.symbols("r_Delta")
eq_r_Delta = sp.Eq(r_Delta, 0.15 * x * ((V_x - V_A) / (V_x + V_A)))
V_xcorr = sp.symbols("V_xcorr")
eq_V_x_corr = sp.Eq(V_xcorr, (V_x - V_A) * r_p / (r_p + r_Delta) + V_A)

# ____________________________________________________________________
# Solutions

eq_V_xR_3dof = eq_V_xR.subs(
    [
        (p, 0),
        (q, 0),
        (w, 0),
    ]
)

eq_V_yR_3dof = eq_V_yR.subs(
    [
        (p, 0),
        (q, 0),
        (w, 0),
    ]
)

eq_V_zR_3dof = eq_V_zR.subs(
    [
        (q, 0),
        (w, 0),
    ]
)

## Lift:
eqs = [
    eq_L,
    eq_C_L,
    eq_Lambda,
    eq_Lambda_g,
    eq_V_R,
    eq_V_xR_3dof,
    eq_V_yR_3dof,
    eq_V_zR_3dof,
    eq_gamma,
]
solution_lift = sp.solve(
    eqs, L, C_L, Lambda, Lambda_g, V_R, V_xR, V_yR, V_zR, gamma, dict=True
)[0]

# solution_lift[Y_R] = sp.Eq(Y_R, n_prop * solution_lift[L]).rhs
lambdas_lift = {key: lambdify(expression) for key, expression in solution_lift.items()}

## Lift no propeller:
solution_no_propeller = deepcopy(solution_lift)
eq_V_x_no_propeller = sp.Eq(V_x, u * (1 - w_f))
solution_no_propeller[L] = sp.simplify(
    solution_lift[L].subs(V_x, eq_V_x_no_propeller.rhs)
)
solution_no_propeller[V_x] = eq_V_x_no_propeller.rhs
solution_no_propeller[C_L] = sp.simplify(
    solution_lift[C_L].subs(V_x, eq_V_x_no_propeller.rhs)
)

solution_no_propeller[Y_R] = sp.Eq(Y_R, n_prop * solution_no_propeller[L]).rhs
lambdas_no_propeller = {
    key: lambdify(expression) for key, expression in solution_no_propeller.items()
}

## Drag:
eqs = [
    eq_D,
    eq_C_D,
    eq_C_D0,
    eq_Lambda,
    eq_Lambda_g,
    eq_CF,
    eq_Re,
    eq_c,
    eq_V_R,
    eq_V_xR_3dof,
    eq_V_yR_3dof,
    eq_V_zR_3dof,
]
solution_drag = sp.solve(
    eqs, D, C_D, C_D0, Lambda, Lambda_g, C_F, R_e, c, V_R, V_xR, V_yR, V_zR, dict=True
)[0]
# solution_drag[X_R] = sp.Eq(X_R, n_prop * solution_drag[D]).rhs

## Propeller influence (to get V_x behind propeller)
eqs = [
    eq_V_x,
    eq_r,
    eq_r_inf,
]
solution_propeller = sp.solve(eqs, V_x, r_p, r_inf, dict=True)[0]
solution_propeller[V_x] = sp.simplify(solution_propeller[V_x].subs(V_inf, eq_V_inf.rhs))
solution_propeller[V_x] = sp.simplify(
    solution_propeller[V_x].subs(
        [
            (C_Th, eq_C_Th.rhs),
        ]
    )
)
solution_propeller[V_x] = solution_propeller[V_x].subs(
    [
        (V_A, eq_V_A.rhs),
    ]
)
solution_propeller[V_xcorr] = eq_V_x_corr.rhs.subs(
    [
        (r_Delta, eq_r_Delta.rhs),
        (r_p, eq_r.rhs),
        (r_inf, solution_propeller[r_inf]),
        (V_inf, eq_V_inf.rhs),
        (C_Th, eq_C_Th.rhs),
        (V_A, eq_V_A.rhs),
        # (V_x, solution_propeller[V_x]),
    ]
)

lambdas_propeller = {
    key: lambdify(expression) for key, expression in solution_propeller.items()
}


# class SemiempiricalRudderSystem(SubSystem):
#    def calculate_forces(self, states_dict: dict, control: dict, calculation: dict):
#
#        self.calculate_propeller_induced_velocity(
#            states_dict=states_dict, control=control, calculation=calculation
#        )
#
#        return calculation
#
#    def calculate_propeller_induced_velocity(
#        self, states_dict: dict, control: dict, calculation: dict
#    ):
#
#        calculation["V_x"] = run(
#            function=lambdas_propeller[V_x],
#            inputs=states_dict,
#            **self.ship.ship_parameters,
#            **control,
#        )
#
#        calculation["X_R"] = 0
#
#        calculation["Y_R"] = run(
#            function=lambdas_lift[Y_R],
#            inputs=states_dict,
#            C_L_tune=self.ship.parameters["C_L_tune"],
#            delta_lim=self.ship.parameters["delta_lim"],
#            kappa=self.ship.parameters["kappa"],
#            **self.ship.ship_parameters,
#            **control,
#            **calculation,
#        )
#
#        calculation["N_R"] = calculation["Y_R"] * self.ship.ship_parameters["x_R"]

from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator

D_F, L_F, alfa_F = sp.symbols(
    "D_F,L_F,alfa_F"
)  # Forces in flow direction (alfa_F=kappa*gamma)


class SemiempiricalRudderSystem(EquationSubSystem):
    def __init__(self, ship: ModularVesselSimulator, create_jacobians=True):
        f_C_L = sp.Function("C_L")(v, r)
        f_V_x = sp.Function("V_x")(u, v, r, thrust)
        subs = [(V_x, f_V_x), (C_L, f_C_L)]

        equations = [
            sp.Eq(V_x, solution_propeller[V_x].subs(subs)),
            sp.Eq(C_L, solution_lift[C_L].subs(subs)),
            sp.Eq(alfa_F, kappa * solution_lift[gamma]),
            sp.Eq(
                D_F, solution_drag[D].subs(subs)
            ),  # renaming of drag to drag in flow direction
            sp.Eq(
                L_F, solution_lift[L].subs(subs)
            ),  # renaming of lift to lift in flow direction
            sp.Eq(X_R, -n_rudd * (-L_F * sp.sin(alfa_F) + D_F * sp.cos(alfa_F))),
            sp.Eq(Y_R, n_rudd * (L_F * sp.cos(alfa_F) + D_F * sp.sin(alfa_F))),
            sp.Eq(N_R, x_R * Y_R),
        ]

        super().__init__(
            ship=ship, equations=equations, create_jacobians=create_jacobians
        )


class SemiempiricalRudderWithoutPropellerInducedSpeedSystem(EquationSubSystem):
    def __init__(self, ship: ModularVesselSimulator, create_jacobians=True):
        f_C_L = sp.Function("C_L")(v, r)
        f_V_x = sp.Function("V_x")(u, v, r, thrust)
        subs = [(V_x, f_V_x), (C_L, f_C_L)]

        # equations = [
        #    # sp.Eq(V_x, u * (1 - w_f)),  # No propeller induced speed here!
        #    sp.Eq(V_x, u),  # No propeller induced speed here!
        #    sp.Eq(C_L, solution_lift[C_L].subs(subs)),
        #    sp.Eq(X_R, -solution_drag[X_R].subs(subs)),
        #    sp.Eq(Y_R, solution_lift[Y_R].subs(subs)),
        #    sp.Eq(N_R, x_R * solution_lift[Y_R]).subs(subs),
        # ]

        equations = [
            sp.Eq(V_x, u * (1 - w_f)),  # No propeller induced speed here!
            # sp.Eq(V_x, u),  # No propeller induced speed here!
            sp.Eq(C_L, solution_lift[C_L].subs(subs)),
            sp.Eq(alfa_F, kappa * solution_lift[gamma]),
            sp.Eq(
                D_F, solution_drag[D].subs(subs)
            ),  # renaming of drag to drag in flow direction
            sp.Eq(
                L_F, solution_lift[L].subs(subs)
            ),  # renaming of lift to lift in flow direction
            sp.Eq(X_R, -n_rudd * (-L_F * sp.sin(alfa_F) + D_F * sp.cos(alfa_F))),
            sp.Eq(Y_R, n_rudd * (L_F * sp.cos(alfa_F) + D_F * sp.sin(alfa_F))),
            sp.Eq(N_R, x_R * Y_R),
        ]

        super().__init__(
            ship=ship, equations=equations, create_jacobians=create_jacobians
        )


a_H, x_H, X_RHI, Y_RHI, N_RHI = sp.symbols("a_H,x_H,X_RHI,Y_RHI,N_RHI")


class RudderHullInteractionSystem(EquationSubSystem):
    def __init__(self, ship: ModularVesselSimulator, create_jacobians=True):
        equations = [
            sp.Eq(X_RHI, 0),
            sp.Eq(Y_RHI, a_H * Y_R),
            sp.Eq(N_RHI, *x_H * a_H * Y_R),
        ]

        super().__init__(
            ship=ship, equations=equations, create_jacobians=create_jacobians
        )

class RudderHullInteractionDummySystem(EquationSubSystem):
    def __init__(self, ship: ModularVesselSimulator, create_jacobians=True):
        equations = [
            sp.Eq(X_RHI, 0),
            sp.Eq(Y_RHI, 0),
            sp.Eq(N_RHI, 0),
        ]

        super().__init__(
            ship=ship, equations=equations, create_jacobians=create_jacobians
        )
