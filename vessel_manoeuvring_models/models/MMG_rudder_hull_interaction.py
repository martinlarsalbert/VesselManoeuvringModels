import sympy as sp
from vessel_manoeuvring_models.symbols import *
from sympy import Eq, symbols, pi, Piecewise
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
from vessel_manoeuvring_models.models.subsystem import EquationSubSystem
from vessel_manoeuvring_models.symbols import *

class RudderHullInteractionSystem(EquationSubSystem):
    def __init__(self, ship: ModularVesselSimulator, create_jacobians=True):
        equations = [
            sp.Eq(X_RHI, 0),
            Eq(Y_RHI, a_H * Y_R),
            Eq(N_RHI, a_H * x_H * Y_R),
        ]

        super().__init__(
            ship=ship, equations=equations, create_jacobians=create_jacobians
        )