from scipy.integrate import solve_ivp
import sympy as sp
from vessel_manoeuvring_models.symbols import *
import dill
from sympy.printing import pretty

from vessel_manoeuvring_models.parameters import df_parameters
from vessel_manoeuvring_models.prime_system import PrimeSystem

p = df_parameters["symbol"]


class ModularVesselSimulator:
    def __init__(
        self,
        X_eq: sp.Eq,
        Y_eq: sp.Eq,
        N_eq: sp.Eq,
        parameters: dict,
        ship_parameters: dict,
        control_keys: list,
    ):
        """Top level of a modular simulation, where the equations define
        the equation of motion as function of subcomponents:
        X_H : Hull
        X_R : Rudder
        X_P : Propeller
        etc.
        giving the top level equation:
        m*(\dot{u} - r**2*x_G - r*v) = X_{\dot{u}}*\dot{u} + X_H + X_R + X_P
        It is up to subcomponents to calculate X_H, X_R etc.

        Parameters
        ----------
        X_eq : sp.Eq
            Equation in X-direction
        Y_eq : sp.Eq
            Equation in Y-direction
        N_eq : sp.Eq
            Equation in N-direction

        """

        self.X_eq = X_eq.copy()
        self.Y_eq = Y_eq.copy()
        self.N_eq = N_eq.copy()

        self.parameters = parameters
        self.ship_parameters = ship_parameters
        self.control_keys = control_keys
        self.prime_system = PrimeSystem(**ship_parameters)
        self.ship_parameters_prime = self.prime_system.prime(ship_parameters)

        self.X_D_eq = sp.Eq(
            X_D_, self.X_eq.subs([(m, 0), (I_z, 0), (u1d, 0), (v1d, 0), (r1d, 0)]).rhs
        )
        self.Y_D_eq = sp.Eq(
            Y_D_, self.Y_eq.subs([(m, 0), (I_z, 0), (u1d, 0), (v1d, 0), (r1d, 0)]).rhs
        )
        self.N_D_eq = sp.Eq(
            N_D_, self.N_eq.subs([(m, 0), (I_z, 0), (u1d, 0), (v1d, 0), (r1d, 0)]).rhs
        )
        self.define_EOM()
        self.subsystems = {}

    def __repr__(self):

        s = (
            f"\n X_eq: \n {pretty(self.X_eq, use_unicode=False)} \n"
            + f"\n Y: \n {pretty(self.Y_eq, use_unicode=False)} \n"
            + f"\n N: \n {pretty(self.N_eq, use_unicode=False)} \n"
        )
        return s

    def save(self, path: str):
        """Save model to pickle file

        Parameters
        ----------
        path : str
            Ex:'model.pkl'
        """

        with open(path, mode="wb") as file:
            dill.dump(self, file=file)

    def __getstate__(self):
        def should_pickle(k):
            return not k in [
                "acceleration_lambda",
                "_X_qs_lambda",
                "_Y_qs_lambda",
                "_N_qs_lambda",
            ]

        return {k: v for (k, v) in self.__dict__.items() if should_pickle(k)}

    @classmethod
    def load(cls, path: str):

        with open(path, mode="rb") as file:
            obj = dill.load(file=file)

        return obj

    def define_EOM(self):
        """Define equation of motion

        Args:
            X_eq (sp.Eq): [description]
            Y_eq (sp.Eq): [description]
            N_eq (sp.Eq): [description]
        """

        X_eq_X_D = self.X_eq.subs(self.X_D_eq.rhs, X_D_)
        Y_eq_Y_D = self.Y_eq.subs(self.Y_D_eq.rhs, Y_D_)
        N_eq_N_D = self.N_eq.subs(self.N_D_eq.rhs, N_D_)

        A, b = sp.linear_eq_to_matrix([X_eq_X_D, Y_eq_Y_D, N_eq_N_D], [u1d, v1d, r1d])
        self.A = A
        self.b = b
        self.acceleartion_eq = A.inv() * b

        ## Rewrite in SI units:
        keys = [
            "Xvdot",
            "Xrdot",
            "Yudot",
            "Yrdot",
            "Nudot",
            "Nvdot",
        ]
        subs = {
            df_parameters.loc[key, "symbol"]: df_parameters.loc[key, "symbol"]
            * df_parameters.loc[key, "denominator"]
            for key in keys
        }
        self.acceleartion_eq_SI = sp.simplify(self.acceleartion_eq.subs(subs))

        ## Lambdify:
        subs = {value: key for key, value in p.items()}
        self.acceleration_lambda_SI = lambdify(self.acceleartion_eq_SI.subs(subs))

        return self.acceleration_lambda_SI

    def calculate_forces(self, states_dict: dict, control: dict):

        calculation = {}
        for name, subsystem in self.subsystems.items():
            subsystem.calculate_forces(
                states_dict=states_dict, control=control, calculation=calculation
            )
        return calculation
