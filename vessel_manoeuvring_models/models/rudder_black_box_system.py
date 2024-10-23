from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
from vessel_manoeuvring_models.models.subsystem import PrimeEquationSubSystem
from copy import deepcopy
from vessel_manoeuvring_models.substitute_dynamic_symbols import lambdify, run
from vessel_manoeuvring_models import equation_helpers

import sympy as sp
from vessel_manoeuvring_models import symbols
from vessel_manoeuvring_models.symbols import *

class PolynomialRudderSystem(PrimeEquationSubSystem):
    import vessel_manoeuvring_models.models.rudder_black_box
    #from vessel_manoeuvring_models.models.semiempirical_covered import *
    
    module = vessel_manoeuvring_models.models.rudder_black_box
    def __init__(
        self,
        ship: ModularVesselSimulator,
        create_jacobians=True,
        in_propeller_race=True,
        suffix="port",
    ):
        self.suffix = suffix
        suffix_str = f"_{suffix}" if len(suffix) > 0 else ""

        equations = {
            value.lhs: value.copy()
            for key, value in self.module.__dict__.items()
            if isinstance(value, sp.Eq)
        }
        equations = {
            key: value for key, value in equations.items() if isinstance(key, sp.Symbol)
        }
        
        tree = equation_helpers.find_equations(symbols.X_R, equations=equations)
        tree.update(equation_helpers.find_equations(symbols.Y_R, equations=equations))
        tree.update(equation_helpers.find_equations(symbols.N_R, equations=equations))
        eqs_rudder = list(tree.values())
        equation_helpers.sort_equations(eqs_rudder)

        renames = {
            self.module.thrust_propeller: f"thrust{suffix_str}"
        }

        eqs_rudder = [eq.subs(renames) for eq in eqs_rudder]

        if len(suffix) > 0:
            # Adding a suffix to distinguish between port and starboard rudder
            subs = {eq.lhs: f"{eq.lhs}{suffix_str}" for eq in eqs_rudder}
            eqs_rudder = [eq.subs(subs) for eq in eqs_rudder]

        PrimeEquationSubSystem.__init__(self=self,
            ship=ship, equations=eqs_rudder, create_jacobians=create_jacobians
        )

        
    def create_partial_derivatives(self):
        self.partial_derivatives = {}  # No partial derivatives from the rudder so far...
        self.partial_derivative_lambdas = {}
        
    
       
    def __getstate__(self):
              
        def should_pickle(k):
            return not k in [
                "lambdas",
            ]
        
        to_pickle = {k: v for (k, v) in self.__dict__.items() if should_pickle(k)}
        return to_pickle
    
    def __setstate__(self,state):
        self.__dict__.update(state)
        #self.equations = d["equations"]
        #self.suffix = d["suffix"]
        self.create_lambdas()


