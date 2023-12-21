import sympy as sp
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
from vessel_manoeuvring_models.models.subsystem import EquationSubSystem
from vessel_manoeuvring_models.symbols import *
a_H, x_H, X_RHI, Y_RHI, N_RHI, Y_R = sp.symbols("a_H,x_H,X_RHI,Y_RHI,N_RHI,Y_R")


class RudderHullInteractionSystem(EquationSubSystem):
    def __init__(self, ship: ModularVesselSimulator, create_jacobians=True):
        equations = [
            sp.Eq(X_RHI, 0),
            sp.Eq(Y_RHI, a_H * Y_R),
            sp.Eq(N_RHI, x_H * N_R),
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