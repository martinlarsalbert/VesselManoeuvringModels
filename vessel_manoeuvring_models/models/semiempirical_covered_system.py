from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
from vessel_manoeuvring_models.models.subsystem import EquationSubSystem
from copy import deepcopy
from vessel_manoeuvring_models.substitute_dynamic_symbols import lambdify, run
from vessel_manoeuvring_models import equation_helpers
import vessel_manoeuvring_models.models.semiempirical_covered
from vessel_manoeuvring_models.models.semiempirical_covered import *
import sympy as sp
from vessel_manoeuvring_models import symbols


class SemiempiricalRudderSystemCovered(EquationSubSystem):
    semiempirical_rudder_module = vessel_manoeuvring_models.models.semiempirical_covered
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
            value.lhs: value
            for key, value in self.semiempirical_rudder_module.__dict__.items()
            if isinstance(value, sp.Eq)
        }
        equations = {
            key: value for key, value in equations.items() if isinstance(key, sp.Symbol)
        }
        equations.pop(symbols.nu)
        tree = equation_helpers.find_equations(symbols.X_R, equations=equations)
        tree.update(equation_helpers.find_equations(symbols.Y_R, equations=equations))
        tree.update(equation_helpers.find_equations(symbols.N_R, equations=equations))
        eqs_rudder = list(tree.values())
        equation_helpers.sort_equations(eqs_rudder)

        renames = {
            Lambda: "lambda_",
            # sp.Derivative(C_L, alpha): "dC_L",
            C_L_no_stall: "C_L_no_stall",
            C_D_C_no_stall: "C_D_C_no_stall",
            C_D_U_no_stall: "C_D_U_no_stall",
            C_L_stall: "C_L_stall",
            C_D_stall_C: "C_D_stall_C",
            C_D_stall_U: "C_D_stall_U",
            C_L_max: "C_L_max",
            C_D_max_C: "C_D_max_C",
            C_D_max_U: "C_D_max_U",
            V_R_x_C: "V_R_x_C",
            V_R_x_U: "V_R_x_U",
            V_R_y: "V_R_y",
            V_infty: "V_infty",
            r_infty: "r_infty",
            V_x_corr: "V_x_corr",
            delta_alpha_s: "delta_alpha_s",
            thrust_propeller: f"thrust{suffix_str}",
            gamma_0: f"gamma_0{suffix_str}",
            # kappa: f"kappa{suffix_str}",
            p: 0,  # no roll velocity
            q: 0,  # no pitch velocity
        }

        eqs_rudder = [eq.subs(renames) for eq in eqs_rudder]

        if len(suffix) > 0:
            # Adding a suffix to distinguish between port and starboard rudder
            subs = {eq.lhs: f"{eq.lhs}{suffix_str}" for eq in eqs_rudder}
            eqs_rudder = [eq.subs(subs) for eq in eqs_rudder]

        EquationSubSystem.__init__(self=self,
            ship=ship, equations=eqs_rudder, create_jacobians=create_jacobians
        )

    def create_lambdas(self, renames={}):
        EquationSubSystem.create_lambdas(self=self, renames=renames)

        if self.suffix == "port":
            self.lambdas["kappa_port"] = self.kappa_port
        elif self.suffix == "stbd":
            self.lambdas["kappa_stbd"] = self.kappa_stbd
        elif self.suffix == "":
            self.lambdas.pop("kappa")  # kappa is a parameter
        else:
            raise ValueError(f"Cannot use suffix:{self.suffix}")
        
    def create_partial_derivatives(self):
        self.partial_derivatives = {}  # No partial derivatives from the rudder so far...
        self.partial_derivative_lambdas = {}
        
    
    @staticmethod
    def kappa_port(gamma_port, kappa_inner, kappa_outer, y_R_port, **kwargs):
        return kappa_func(
            gamma=gamma_port,
            kappa_inner=kappa_inner,
            kappa_outer=kappa_outer,
            y_R=y_R_port,
        )

    @staticmethod
    def kappa_stbd(gamma_stbd, kappa_inner, kappa_outer, y_R_stbd, **kwargs):
        return kappa_func(
            gamma=gamma_stbd,
            kappa_inner=kappa_inner,
            kappa_outer=kappa_outer,
            y_R=y_R_stbd,
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


def kappa_func(gamma, kappa_inner, kappa_outer, y_R, **kwargs):
    condition = ((gamma >= 0) & (y_R <= 0)) | ((gamma < 0) & (y_R > 0))
    return np.where(condition, kappa_outer, kappa_inner)
