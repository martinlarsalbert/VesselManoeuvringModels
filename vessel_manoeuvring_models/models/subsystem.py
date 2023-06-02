import sympy as sp
import numpy as np
from vessel_manoeuvring_models.parameters import df_parameters

p = df_parameters["symbol"]
from vessel_manoeuvring_models.substitute_dynamic_symbols import lambdify, run
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
from vessel_manoeuvring_models.prime_system import standard_units


class SubSystem:
    def __init__(self, ship: ModularVesselSimulator):
        self.ship = ship

    def calculate_forces(self, states_dict: dict, control: dict, calculation: dict):
        return calculation


class PrimeEquationSystem(SubSystem):
    def __init__(self, ship: ModularVesselSimulator, equations=[]):
        super().__init__(ship=ship)
        self.equations = {str(eq.lhs): eq for eq in equations}
        self.create_lambdas()

    def create_lambdas(self):
        subs = {value: key for key, value in p.items()}
        self.lambdas = {}
        for name, eq in self.equations.items():
            self.lambdas[name] = lambdify(eq.rhs.subs(subs))

    def calculate_forces(self, states_dict: dict, control: dict, calculation: dict):

        prime_system = self.ship.prime_system
        U = np.sqrt(states_dict["u"] ** 2 + states_dict["v"] ** 2)
        states_dict_prime = prime_system.prime(states_dict, U=U)
        control_prime = prime_system.prime(control, U=U)
        calculation_prime = prime_system.prime(calculation, U=U)

        for key, lambda_ in self.lambdas.items():
            assert not key in calculation, f"{key} has already been calculated"
            unit = standard_units[key]

            result_prime = run(
                function=lambda_,
                inputs=states_dict_prime,
                **control_prime,
                **calculation_prime,
                **self.ship.ship_parameters_prime,
                **self.ship.parameters,
            )

            result_SI = prime_system._unprime(result_prime, U=U, unit=unit)
            calculation[key] = result_SI

        return calculation
