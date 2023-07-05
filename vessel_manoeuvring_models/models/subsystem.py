import sympy as sp
import numpy as np
from vessel_manoeuvring_models.parameters import df_parameters

from vessel_manoeuvring_models.substitute_dynamic_symbols import lambdify, run
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
from vessel_manoeuvring_models.prime_system import standard_units
from vessel_manoeuvring_models.symbols import *

p = df_parameters["symbol"]
subs_simpler = {value: key for key, value in p.items()}
subs_simpler[psi] = "psi"
from vessel_manoeuvring_models import prime_system
from copy import deepcopy


class SubSystem:
    def __init__(
        self,
        ship: ModularVesselSimulator,
        create_jacobians=True,
    ):
        self.ship = ship
        self.create_jacobians = create_jacobians
        if self.create_jacobians:
            self.create_partial_derivatives()

    def copy_and_refer_other_ship(self, ship: ModularVesselSimulator):
        copy = deepcopy(self)
        copy.ship = ship
        return copy

    def calculate_forces(self, states_dict: dict, control: dict, calculation: dict):
        return calculation

    def create_partial_derivatives(self):
        self.partial_derivative_lambdas = {}

    def calculate_partial_derivatives(
        self, states_dict: dict, control: dict, calculation: dict
    ):

        states_dict["U"] = np.sqrt(states_dict["u"] ** 2 + states_dict["v"] ** 2)
        for key, partial_derivative_lambda in self.partial_derivative_lambdas.items():
            assert (
                not key in calculation
            ), f"Partial derivative {key} has already been calculated"

            try:
                result = run(
                    function=partial_derivative_lambda,
                    inputs=states_dict,
                    **control,
                    **calculation,
                    **self.ship.ship_parameters,
                    **self.ship.parameters,
                )
            except Exception as e:
                raise ValueError(f"Failed to calculate {key}")

            calculation[key] = result


class EquationSubSystem(SubSystem):
    def __init__(
        self, ship: ModularVesselSimulator, create_jacobians=True, equations=[]
    ):
        self.equations = {str(eq.lhs.name): eq for eq in equations}
        super().__init__(ship=ship, create_jacobians=create_jacobians)
        self.create_lambdas()

    def create_lambdas(self):
        self.lambdas = {}
        for name, eq in self.equations.items():
            self.lambdas[name] = lambdify(
                eq.rhs.subs(subs_simpler), substitute_functions=True
            )

    def create_partial_derivatives(self):
        self.partial_derivatives = {}

        self.get_partial_derivatives()

        self.partial_derivative_lambdas = {
            key: lambdify(value, substitute_functions=True)
            for key, value in self.partial_derivatives.items()
        }

    def get_partial_derivatives(self):
        for name, eq in self.equations.items():
            self.partial_derivatives.update(self.get_eq_partial_derivatives(eq=eq))

    def get_eq_partial_derivatives(self, eq):
        return {
            "dd"
            + str(state).lower().replace("\\", "")
            + str(eq.lhs): eq.rhs.diff(state).subs(subs_simpler)
            for state in self.ship.states
        }

    def calculate_forces(self, states_dict: dict, control: dict, calculation: dict):
        """Calculate forces from system

        Parameters
        ----------
        states_dict : dict
            states in SI units!
        control : dict
            control in SI units!
        calculation : dict
            results from previous calculations that can be used as input to this one.

        Returns
        -------
        dict
            calculation dict updated with the forces from this system
        """

        states_dict["U"] = np.sqrt(states_dict["u"] ** 2 + states_dict["v"] ** 2)

        for key, lambda_ in self.lambdas.items():
            assert not key in calculation, f"{key} has already been calculated"

            try:
                result_SI = run(
                    function=lambda_,
                    inputs=states_dict,
                    **control,
                    **calculation,
                    **self.ship.ship_parameters,
                    **self.ship.parameters,
                )
            except Exception as e:
                raise ValueError(f"Failed to calculate {key}")

            calculation[key] = result_SI

        return calculation


class PrimeEquationSubSystem(EquationSubSystem):
    def calculate_forces(
        self, states_dict: dict, control: dict, calculation: dict
    ) -> dict:
        """Calculate forces from system

        Parameters
        ----------
        states_dict : dict
            states in SI units!
        control : dict
            control in SI units!
        calculation : dict
            results from previous calculations that can be used as input to this one.

        Returns
        -------
        dict
            calculation dict updated with the forces from this system
        """

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

    def get_partial_derivatives(self):

        u_prime, v_prime = sp.symbols("u' v'")
        subs_prime = [
            (m, m / prime_system.df_prime.mass.denominator),
            (I_z, I_z / prime_system.df_prime.inertia_moment.denominator),
            (x_G, x_G / prime_system.df_prime.length.denominator),
            (u_prime, u / sp.sqrt(u**2 + v**2)),
            (v_prime, v / sp.sqrt(u**2 + v**2)),
            (r, r / (sp.sqrt(u**2 + v**2) / L)),
            (thrust, thrust / (sp.Rational(1, 2) * rho * (u**2 + v**2) * L**2)),
        ]

        for name, eq in self.equations.items():
            unit = standard_units[name]
            denominator = prime_system.df_prime.loc["denominator", unit]
            denominator = denominator.subs(U, sp.sqrt(u**2 + v**2))
            eq_ = eq.subs(
                [
                    (
                        u,
                        u_prime,
                    ),  # u_prime and u as a denominator needs to be distinguished,
                    # so that the denominator is not applied twice for r etc.
                    (v, v_prime),
                ]
            )
            eq_SI = sp.Eq(eq.lhs, sp.simplify(eq_.rhs.subs(subs_prime) * denominator))
            self.partial_derivatives.update(self.get_eq_partial_derivatives(eq=eq_SI))
