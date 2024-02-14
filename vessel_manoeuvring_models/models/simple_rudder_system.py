from sympy import Eq, symbols, Symbol, cos, sin, Derivative, atan, Piecewise, pi
from vessel_manoeuvring_models.models.subsystem import PrimeEquationSubSystem
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
from vessel_manoeuvring_models.symbols import *
from vessel_manoeuvring_models.parameters import df_parameters

p = df_parameters["symbol"]

eq_X_R = Eq(
    X_R,
    p.Xdeltadelta * delta ** 2
)

eq_Y_R = Eq(
    Y_R,
    p.Ydelta * delta
    + p.Ythrustdelta * thrust * delta
    + p.Ythrust * thrust
    + p.Yvvdelta * v ** 2 * delta
)
eq_N_R = Eq(
    N_R,
    p.Ndelta * delta
    + p.Nthrustdelta * thrust * delta
    + p.Nthrust * thrust
    + p.Nvvdelta * v ** 2 * delta
)


class SimpleRudderSystem(PrimeEquationSubSystem):
    def __init__(self, ship: ModularVesselSimulator, create_jacobians=True, **kwargs):
        eqs_rudder = [
            eq_X_R,
            eq_Y_R,
            eq_N_R,
        ]

        super().__init__(
            ship=ship, equations=eqs_rudder, create_jacobians=create_jacobians
        )
