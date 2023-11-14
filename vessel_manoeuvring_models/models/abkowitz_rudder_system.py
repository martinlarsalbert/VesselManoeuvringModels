from sympy import Eq, symbols, Symbol, cos, sin, Derivative, atan, Piecewise, pi
from vessel_manoeuvring_models.models.subsystem import PrimeEquationSubSystem
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
from vessel_manoeuvring_models.symbols import *
from vessel_manoeuvring_models.parameters import df_parameters

p = df_parameters["symbol"]

eq_X_R = Eq(
    X_R,
    # p.Xrdelta * r * delta
    # + p.Xurdelta * u * r * delta
    p.Xdeltadelta * delta**2
    # + p.Xudeltadelta * u * delta**2
    # + p.Xvdelta * v * delta
    # + p.Xuvdelta * u * v * delta,
)

eq_Y_R = Eq(
    Y_R,
    p.Ydelta * delta + p.Ydeltadeltadelta * delta**3
    # + p.Yudelta * u * delta
    # + p.Yuudelta * u**2 * delta
    + p.Yvdeltadelta * v * delta**2
    + p.Yvvdelta * v**2 * delta
    + p.Yrdeltadelta * r * delta**2
    + p.Yrrdelta * r**2 * delta
    + p.Yvrdelta * v * r * delta
    # + p.Ythrustdelta * thrust * delta,
)
eq_N_R = Eq(
    N_R,
    p.Ndelta * delta + p.Ndeltadeltadelta * delta**3
    # + p.Nudelta * u * delta
    # + p.Nuudelta * u**2 * delta
    + p.Nrrdelta * r**2 * delta
    + p.Nvrdelta * v * r * delta
    + p.Nvdeltadelta * v * delta**2
    + p.Nrdeltadelta * r * delta**2
    + p.Nvvdelta * v**2 * delta
    # + p.Nthrustdelta * thrust * delta,
)


class AbkowitzRudderSystem(PrimeEquationSubSystem):
    def __init__(self, ship: ModularVesselSimulator, create_jacobians=True, **kwargs):
        eqs_rudder = [
            eq_X_R,
            eq_Y_R,
            eq_N_R,
        ]

        super().__init__(
            ship=ship, equations=eqs_rudder, create_jacobians=create_jacobians
        )
