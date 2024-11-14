import sympy as sp
from vessel_manoeuvring_models.symbols import *
from sympy import symbols, Eq
from vessel_manoeuvring_models.models.subsystem import EquationSubSystem
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator

X_RM,Y_RM = symbols("X_RM,Y_RM")  # Measured forces

class RudderSimpleSystem(EquationSubSystem):
    def __init__(self, ship: ModularVesselSimulator, create_jacobians=True):
        from vessel_manoeuvring_models.parameters import df_parameters

        eq_X_R = Eq(X_R, X_RM)
        eq_Y_R = Eq(Y_R, Y_RM)
        eq_N_R = Eq(N_R, Y_RM*x_r)
                
        equations = [
            eq_X_R,
            eq_Y_R,
            eq_N_R,
        ]
        super().__init__(
            ship=ship, equations=equations, create_jacobians=False
        )