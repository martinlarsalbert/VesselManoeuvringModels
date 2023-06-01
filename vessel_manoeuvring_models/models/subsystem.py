import sympy as sp
import numpy as np
from vessel_manoeuvring_models.parameters import df_parameters

p = df_parameters["symbol"]
from vessel_manoeuvring_models.substitute_dynamic_symbols import lambdify, run
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator


class SubSystem:
    def __init__(self, ship: ModularVesselSimulator):
        self.ship = ship

    def calculate_forces(self, states_dict: dict, control: dict, calculation: dict):
        return calculation


class PrimeEquationSystem(SubSystem):
    def __init__(
        self,
        ship: ModularVesselSimulator,
        X_eq: sp.Eq,
        Y_eq: sp.Eq,
        N_eq: sp.Eq,
    ):
        super().__init__(ship=ship)
        self.X_eq = X_eq.copy()
        self.Y_eq = Y_eq.copy()
        self.N_eq = N_eq.copy()
        self.create_lambdas()

    def create_lambdas(self):
        subs = {value: key for key, value in p.items()}
        self.lambda_X = lambdify(self.X_eq.rhs.subs(subs))
        self.lambda_Y = lambdify(self.Y_eq.rhs.subs(subs))
        self.lambda_N = lambdify(self.N_eq.rhs.subs(subs))

    def calculate_forces(self, states_dict: dict, control: dict, calculation: dict):

        equations = [
            (self.X_eq, self.lambda_X, "force"),
            (self.Y_eq, self.lambda_Y, "force"),
            (self.N_eq, self.lambda_N, "moment"),
        ]
        prime_system = self.ship.prime_system
        U = np.sqrt(states_dict["u"] ** 2 + states_dict["v"] ** 2)
        states_dict_prime = prime_system.prime(states_dict, U=U)
        control_prime = prime_system.prime(control, U=U)
        calculation_prime = prime_system.prime(calculation, U=U)

        for item in equations:
            eq = item[0]
            key = str(eq.lhs)
            assert not key in calculation, f"{key} has already been calculated"
            lambda_ = item[1]
            unit = item[2]

            result_prime = run(
                function=lambda_,
                inputs=states_dict_prime,
                **control_prime,
                **calculation_prime,
                **self.ship.ship_parameters_prime,
                **self.ship.parameters,
            )

            result_SI = prime_system._unprime(result_prime, U=U, unit=unit)
            key = str(eq.lhs)
            assert not key in calculation, f"{key} has already been"
            calculation[key] = result_SI

        return calculation
