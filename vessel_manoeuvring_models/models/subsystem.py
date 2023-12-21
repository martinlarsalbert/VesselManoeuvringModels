import sympy as sp
import numpy as np
from vessel_manoeuvring_models.parameters import df_parameters

from vessel_manoeuvring_models.substitute_dynamic_symbols import lambdify, run, equation_to_python_method, expression_to_python_method
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
from vessel_manoeuvring_models.prime_system import standard_units
from vessel_manoeuvring_models.symbols import *
from vessel_manoeuvring_models.models.diff_eq_to_matrix import DiffEqToMatrix

p = df_parameters["symbol"]
subs_simpler = {value: key for key, value in p.items()}
subs_simpler[psi] = "psi"
from vessel_manoeuvring_models import prime_system
from copy import deepcopy
import logging

log = logging.getLogger(__name__)

class SubSystem:
    def __init__(
        self,
        ship: ModularVesselSimulator,
        create_jacobians=True,
    ):
        """Sub system to a ModularVesselSimulator

        Parameters
        ----------
        ship : ModularVesselSimulator
            _description_
        create_jacobians : bool, optional
            _description_, by default True
        """
        self.ship = ship
        self.create_jacobians = create_jacobians
        if self.create_jacobians:
            self.create_partial_derivatives()

    def copy_and_refer_other_ship(self, ship: ModularVesselSimulator):
        copy = deepcopy(self)
        copy.ship = ship
        return copy

    def calculate_forces(
        self,
        states_dict: dict,
        control: dict,
        calculation: dict,
        allow_double_calc=False,
    ):
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
                result = partial_derivative_lambda(
                    **states_dict,
                    **control,
                    **calculation,
                    **self.ship.ship_parameters,
                    **self.ship.parameters,
                )
                #result = run(
                #    function=partial_derivative_lambda,
                #    inputs=states_dict,
                #    **control,
                #    **calculation,
                #    **self.ship.ship_parameters,
                #    **self.ship.parameters,
                #)
            except Exception as e:
                raise ValueError(f"Failed to calculate {key}")

            calculation[key] = result
            
    def __getstate__(self):
        def should_pickle(k):
            return not k in [
                "ship",
            ]

        return {k: v for (k, v) in self.__dict__.items() if should_pickle(k)}


class EquationSubSystem(SubSystem):
    def __init__(
        self,
        ship: ModularVesselSimulator,
        create_jacobians=True,
        equations=[],
        renames={},
    ):
        """Sub system to a ModularVesselSimulator

        Parameters
        ----------
        ship : ModularVesselSimulator
            _description_
        create_jacobians : bool, optional
            _description_, by default True
        equations : list, optional
            A list of SymPy equations describing this system
        """
        self.equations = {str(eq.lhs.name): eq for eq in equations}
        super().__init__(ship=ship, create_jacobians=create_jacobians)
        self.create_lambdas(renames=renames)

    def create_lambdas(self, renames={}):
        self.lambdas = {}

        renames_all = subs_simpler.copy()
        renames_all.update(renames)

        for name, eq in self.equations.items():
            #self.lambdas[name] = lambdify(
            #    eq.rhs.subs(renames_all), substitute_functions=True
            #)
            self.lambdas[name] = self.equation_to_python_method(eq=eq.subs(renames_all), substitute_functions=True, name=name)

    def create_partial_derivatives(self):
        self.partial_derivatives = {}

        self.get_partial_derivatives()

        self.partial_derivative_lambdas = {
            #key: lambdify(value, substitute_functions=True)
            key: self.expression_to_python_method(value, function_name=key, substitute_functions=True)
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
            for state in self.ship.states_in_jacobi
        }

    def calculate_forces(
        self,
        states_dict: dict,
        control: dict,
        calculation: dict,
        allow_double_calc=False,
    ):
        """Calculate forces from system

        Parameters
        ----------
        states_dict : dict
            states in SI units!
        control : dict
            control in SI units!
        calculation : dict
            results from previous calculations that can be used as input to this one.
        allow_double_calc: bool, default False
            is it allowed that a variable is calculated more than once?

        Returns
        -------
        dict
            calculation dict updated with the forces from this system
        """

        states_dict = states_dict.copy()
        states_dict["U"] = np.sqrt(states_dict["u"] ** 2 + states_dict["v"] ** 2)

        for key, lambda_ in self.lambdas.items():
            if not allow_double_calc:
                assert not key in calculation, f"{key} has already been calculated"

            try:
                result_SI = lambda_(
                    **states_dict,
                    **control,
                    **calculation,
                    **self.ship.ship_parameters,
                    **self.ship.parameters,
                )
            except Exception as e:
                log.error(e)
                try:
                    # slower option:
                    result_SI = run(
                    function=lambda_,
                    inputs=states_dict,
                    **control,
                    **calculation,
                    **self.ship.ship_parameters,
                    **self.ship.parameters,)

            
                except Exception as e:
                    raise ValueError(f"Failed to calculate {key}")

            calculation[key] = result_SI

        return calculation

    def calculate_parameter_contributions(
        self, eq: sp.Eq, data: pd.DataFrame, base_features=[u, v, r]
    ):
        to_matrix = DiffEqToMatrix(eq, label=eq.lhs, base_features=base_features)
        X = to_matrix.calculate_features(data=data)

        parameters = pd.Series(self.ship.parameters)
        columns = list(set(parameters.index) & set(X.columns))
        parameters = parameters[columns].copy()
        df_parameters_contributions = X * parameters
        return df_parameters_contributions

    def equation_to_python_method(self,eq, substitute_functions=False, name=None):
        full_function_name = f"{self.__class__.__name__}_{name}"  # Creating a unique function name to avoid clash with other classes
        return equation_to_python_method(eq=eq, name=full_function_name, substitute_functions=substitute_functions)

    def expression_to_python_method(self,expression, function_name:str, substitute_functions=False):
        full_function_name = f"{self.__class__.__name__}_{function_name}"  # Creating a unique function name to avoid clash with other classes
        return expression_to_python_method(expression=expression, function_name=full_function_name, substitute_functions=substitute_functions)


class PrimeEquationSubSystem(EquationSubSystem):
    def __init__(
        self,
        ship: ModularVesselSimulator,
        Fn0: float = 0,
        create_jacobians=True,
        equations=[],
        g=9.81,
    ):
        """Sub system defined in prime system

        Parameters
        ----------
        ship : ModularVesselSimulator
            _description_
        Fn0: float, default 0
            nominal speed, expressed as froude number (Fn0=U0/sqrt(Lpp*g)), which is used to define: u^ = u - U0
            u^ is a small perturbation that is used insted of the actual surge velocity u (see why below).
            Using nondimensional Fn0 instead of U0, asserts a scalable model.

            If u would be used, u' would be calculated as:
            u'=u/V
            ...On a straight course, where u=V, during a resistance test this means that u'=1 for all speeds!
            This means that a nonlinear resistance model cannot be fitted!
            Ex: X_h = Xu*u' + Xuu*u'**2 would reduce to X_h = Xu + Xuu, which cannot be regressed!
            Setting U0 = min(V) in a captive test is a good choice. U0 = 0 also works,
            but will force the resistance model to be linear, with only one coefficient, as described above.
            The V0 needs to be subtracted from the captive test surge velocity u, during the regression.

        create_jacobians : bool, optional
            _description_, by default True
        equations : list, optional
            A list of SymPy equations describing this system
        g : float, defaul 9.81
            acceleration to calculate Froude number for Fn0
        """

        super().__init__(
            ship=ship, create_jacobians=create_jacobians, equations=equations
        )

        self.create_lambdas()
        self.Fn0 = Fn0
        self.g = g

    @property
    def U0(self):
        if hasattr(self.ship, "Fn0"):
            return self.ship.U0
        else:
            raise ValueError("The U0 must be defined at the ship.")

    @U0.setter
    def U0(self, U0):
        raise ValueError("It is not allowed to define the U0 from the system anymore.")

    def calculate_forces(
        self,
        states_dict: dict,
        control: dict,
        calculation: dict,
        allow_double_calc=False,
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
        allow_double_calc: bool, default False
            is it allowed that a variable is calculated more than once?

        Returns
        -------
        dict
            calculation dict updated with the forces from this system
        """

        states_dict = states_dict.copy()
        control = control.copy()

        prime_system = self.ship.prime_system
        U = np.sqrt(states_dict["u"] ** 2 + states_dict["v"] ** 2)
        states_dict_u = states_dict.copy()

        states_dict_u["u"] -= self.U0

        states_dict_prime = prime_system.prime(states_dict_u, U=U)
        control_prime = prime_system.prime(control, U=U)
        calculation_prime = prime_system.prime(calculation, U=U)

        for key, lambda_ in self.lambdas.items():
            if not allow_double_calc:
                assert not key in calculation, f"{key} has already been calculated"
            unit = standard_units[key]

            try:
                result_prime = lambda_(
                    **states_dict_prime,
                    **control_prime,
                    **calculation_prime,
                    **self.ship.ship_parameters_prime,
                    **self.ship.parameters,
                )
            except:
                # slower option:
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

    def calculate_parameter_contributions(
        self, eq: sp.Eq, data: pd.DataFrame, unit: str, base_features=[u, v, r]
    ):
        to_matrix = DiffEqToMatrix(eq, label=eq.lhs, base_features=base_features)
        data_MDL_prime = self.ship.prime_system.prime(
            data[self.ship.states_str], U=data["V"]
        )
        X = to_matrix.calculate_features(data=data_MDL_prime)

        parameters = pd.Series(self.ship.parameters)
        columns = list(set(parameters.index) & set(X.columns))
        parameters = parameters[columns].copy()
        df_parameters_contributions_prime = X * parameters

        units = {key: unit for key in df_parameters_contributions_prime.columns}
        df_parameters_contributions = self.ship.prime_system.unprime(
            df_parameters_contributions_prime, U=data["V"], units=units
        )

        return df_parameters_contributions
