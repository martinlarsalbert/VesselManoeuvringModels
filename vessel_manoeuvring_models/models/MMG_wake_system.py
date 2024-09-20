from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
from vessel_manoeuvring_models.models.subsystem import EquationSubSystem
from copy import deepcopy
from vessel_manoeuvring_models.substitute_dynamic_symbols import lambdify, run
from vessel_manoeuvring_models import equation_helpers
import sympy as sp
from vessel_manoeuvring_models import symbols
from vessel_manoeuvring_models.symbols import *

class MMGWakeSystem(EquationSubSystem):
    import vessel_manoeuvring_models.models.MMG_wake
    #from vessel_manoeuvring_models.models.MMG_wake import *
    
    module = vessel_manoeuvring_models.models.MMG_wake
    def __init__(
        self,
        ship: ModularVesselSimulator,
        create_jacobians=True,
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
        #equations.pop(symbols.nu)
        #equations.pop(symbols.nu_c)
        #equations.pop(symbols.nu_r)
        #equations.pop(symbols.eta)
        
        tree = equation_helpers.find_equations(self.module.w_f, equations=equations)
        eqs = list(tree.values())
        equation_helpers.sort_equations(eqs)

        renames = {
            p: 0,  # no roll velocity
            q: 0,  # no pitch velocity
        }

        eqs = [eq.subs(renames) for eq in eqs]

        if len(suffix) > 0:
            # Adding a suffix to distinguish between port and starboard rudder
            subs = {eq.lhs: f"{eq.lhs}{suffix_str}" for eq in eqs}
            eqs = [eq.subs(subs) for eq in eqs]

        EquationSubSystem.__init__(self=self,
            ship=ship, equations=eqs, create_jacobians=create_jacobians
        )
    
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