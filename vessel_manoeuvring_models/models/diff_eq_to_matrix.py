import re
import pandas as pd
import numpy as np
import sympy as sp
from vessel_manoeuvring_models.substitute_dynamic_symbols import lambdify, run
from vessel_manoeuvring_models import symbols

from vessel_manoeuvring_models.parameters import df_parameters


class DiffEqToMatrix:
    """This class reformulates a differential equation into a matrix form regression problem:
    y = X*beta + epsilon

    Example:
    Diff eq:
    phi2d + B*phi1d + C*phi = 0

    y      = X     * beta           + epsilon
    -phi.diff().diff() = [B C] x [phi.diff() phi].T  + epsilon

    """

    def __init__(
        self,
        ode: sp.Eq,
        label: sp.Symbol,
        base_features=[],
        exclude_parameters: dict = {},
        exclude_parameters_denominator: float = 1.0,
    ):
        """[summary]

        Parameters
        ----------
        ode : sp.Eq
            ordinary differential equation
        label : sp.Symbol
            label <-> dependent variable in regression (usually acceleration)
        base_features : list, optional
            list with base features, ex: [phi] (derivatives phi.diff() and polynomial combinations such as phi.diff()**3 will be figured out)
        exclude_parameters : dict, optional
            Exclude some parameters from the regression by instead providing their value.
            Ex:
            exclude_parameters = {'Xthrust':0.95}
            means that Xthrust parameter will not be regressed, instead a value of 0.95 will be used.
            Note! the sum of all excluded parameters and their features are substracted from the y (label).
            y = X_exclude*beta_exclude + X*beta
            --> y - X_exclude*beta_exclude = + X*beta
        exclude_parameters_denominator : float
        when parameters are excluded they need to be moved to LHS. And they are divided by this parameter.

        """

        self.ode = ode
        assert isinstance(self.ode, sp.Eq)

        assert label.name != "y", "label name 'y' is not allowed."
        self.label = label
        assert isinstance(self.label, sp.Expr)

        self.base_features = base_features

        self.exclude_parameters = pd.Series(exclude_parameters)

        self.setup()

        self.exclude_parameters_denominator = exclude_parameters_denominator

    def __repr__(self):
        return str(self.ode)

    def setup(self):
        # Swap around equation to get acceleration in left hand side:
        self.get_acceleration()

        # Get a list of hydrodynamic derivatives (coefficients) as sympy symbols:
        self.coefficients = self.get_coefficients()

        # Get the expressions that the coefficients are multiplied with, which will later become the "combined features":
        self.parts = self.get_parts()

        # Express the diff eq. as a regression problem in matrix form:
        self.get_labels_and_features()

    @property
    def X_lambda(self):
        ## If there is a constant in the equation eq_X will have a 1 that cannot go into the X_lambda
        if 1 in self.eq_X.rhs:
            return lambdify(
                sp.matrices.immutable.ImmutableDenseMatrix(
                    self.eq_X.rhs[1:]
                )  # (1 is always the first feature)
            )
        else:
            return lambdify(self.eq_X.rhs)

    @property
    def y_lambda(self):
        p = df_parameters["symbol"]
        subs = {value: key for key, value in p.items()}

        return lambdify(self.eq_y.rhs.subs(subs))

    @property
    def acceleration_lambda(self):
        subs = self.feature_names_subs()
        return lambdify(sp.solve(self.acceleration_equation.subs(subs), self.label)[0])

    def feature_names_subs(self):
        ## Rename:
        columns_raw = list(self.eq_beta.rhs)
        subs = {}

        regexp = re.compile(r"\\dot{([^}])+}")

        def replacer(match):
            return r"%sdot" % match.group(1)

        for symbol in columns_raw:
            ascii_symbol = str(symbol)
            ascii_symbol = regexp.sub(repl=replacer, string=ascii_symbol)
            ascii_symbol = ascii_symbol.replace("_", "")
            ascii_symbol = ascii_symbol.replace("{", "")
            ascii_symbol = ascii_symbol.replace("}", "")
            ascii_symbol = ascii_symbol.replace("\\", "")
            ascii_symbol = ascii_symbol.replace("-", "")  # Little bit dangerous
            subs[symbol] = ascii_symbol

        return subs

    def calculate_features(
        self, data: pd.DataFrame, parameters={}, simplify_names=True
    ):
        X = run(function=self.X_lambda, **data, **parameters)

        try:
            X = X.reshape(X.shape[1], X.shape[-1]).T
        except Exception:
            X = X.reshape(X.shape[0], X.shape[-1]).T

        # If there is a constant in the equation eq_X will have a 1 that needs to adde as the first column manually
        if 1 in self.eq_X.rhs:
            ones = np.ones(shape=(len(data), 1))
            if len(X) > 0:
                X = np.concatenate([ones, X], axis=1)
            else:
                X = ones  # only a constant in rhs

        subs = self.feature_names_subs()
        if simplify_names:
            columns = list(subs.values())
        else:
            columns = [str(key) for key in subs.keys()]

        X = pd.DataFrame(data=X, index=data.index, columns=columns)

        return X

    def calculate_label(self, y: np.ndarray):
        return self.y_lambda(y)

    def calculate_features_and_label(
        self, data: pd.DataFrame, y: np.ndarray, simplify_names=True, parameters={}
    ):
        y = y.copy()
        X = self.calculate_features(
            data=data, simplify_names=simplify_names, parameters=parameters
        )
        y = self.calculate_label(y=y)

        ## Exclude parameters:
        keep = list(set(X.columns) - set(self.exclude_parameters.keys()))
        exclude = list(set(X.columns) & set(self.exclude_parameters.keys()))

        if len(exclude) > 0:
            feature_excludes = X[exclude].copy()
            y_excludes = feature_excludes.multiply(
                self.exclude_parameters[exclude], axis=1
            )
            y_exclude = y_excludes.sum(axis=1)
            X = X[keep].copy()
            y -= y_exclude / self.exclude_parameters_denominator

        # Difference method:
        # X = X.diff().iloc[1:].copy()
        # y = y.diff().iloc[1:].copy()

        return X, y

    def get_acceleration(self):
        """Swap around equation to get acceleration in left hand side"""
        self.acceleration_equation = sp.Eq(
            self.label, sp.solve(self.ode, self.label)[0]
        )

    def get_coefficients(self) -> list:
        """Get a list of hydrodynamic derivatives (coefficients) as sympy symbols.

        Ex:
        [N_{0},N_{0uu},N_{0u},...]

        """
        return get_coefficients(
            eq=self.acceleration_equation, base_features=self.base_features
        )

    def get_parts(self) -> tuple:
        """Get the expressions that the coefficients are multiplied with, which will later become the "combined features"

        Returns
        -------
        Ex:
        (1, delta(t)**3, r(t)**3, ...)
        """

        return self.acceleration_equation.rhs.subs(
            [(c, 1) for c in self.coefficients]
        ).args

    def get_labels_and_features(self):
        """Express the diff eq. as a regression problem in matrix form"""

        self.y_ = sp.symbols("y")

        # Remove excluded parameters
        equation = self.acceleration_equation.copy()
        # if an excluded parameter appears twice... (Xthrust*thrust_port + X_thrust*thrust_stbd)
        # excludes = {key: 0 for key in self.exclude_parameters}
        # equation = equation.subs(excludes)

        ## Extract the coefficients (based on the base features)
        subs = [(feature, 1) for feature in self.base_features]
        parts = equation.rhs.subs(subs)
        if isinstance(parts, sp.Symbol):
            coefficients = [parts]  # If there is only one coefficient
        else:
            coefficients = [coeff for coeff in parts.args]
        check_coefficients(coefficients)

        ## Find the feature expression that is associated with each coefficient,
        # and put it in a list: columns
        subs = {coeff: 0 for coeff in coefficients}
        columns = []
        for coeff in coefficients:
            subs_mask = subs.copy()
            subs_mask[coeff] = 1
            columns.append(equation.rhs.subs(subs_mask))

        if 1 in columns:
            index = columns.index(1)  # Where is the 1? (The constant)
            coefficient_1 = coefficients[index]

            ordered_coefficients = coefficients.copy()
            ordered_coefficients.remove(coefficient_1)
            ordered_coefficients = [
                coefficient_1
            ] + ordered_coefficients  # Put the constant coefficient first

            ordered_columns = columns.copy()
            ordered_columns.remove(1)
            ordered_columns = [1] + ordered_columns  # Put the constant feature first
        else:
            ordered_coefficients = coefficients
            ordered_columns = columns

        ## Feature matrix
        self.X_matrix = sp.Matrix(ordered_columns).T

        ## Put this into some equations:
        self.beta_ = sp.MatrixSymbol("beta", len(ordered_columns), 1)
        self.eq_beta = sp.Eq(self.beta_, sp.Matrix(ordered_coefficients))

        self.X_ = sp.MatrixSymbol("X", 1, len(ordered_columns))
        self.eq_X = sp.Eq(self.X_, self.X_matrix)

        self.eq_y = sp.Eq(self.y_, self.label)


def check_coefficients(coefficients: list):
    for coeff in coefficients:
        ok = True
        if len(coeff.args) == 2:
            if not isinstance(coeff.args[0], sp.core.numbers.Integer):
                ok = False

        elif len(coeff.args) > 2:
            ok = False

        if not ok:
            raise AssertionError(
                f"Cannot find pure coefficients (found: {coeff}), perhaps alter the base_features?"
            )


def get_coefficients(eq, base_features: list) -> list:
    """Get a list of hydrodynamic derivatives (coefficients) as sympy symbols.

    Args:
    eq : sympy equation
    base_features : list of from sympy.physics.mechanics.dynamicsymbols
    Ex: base_features = [u,v,r,delta,thrust]

    Ex:
    [N_{0},N_{0uu},N_{0u},...]

    """

    coefficients = []

    # Propose derivatives:
    derivatives = []
    for base_feature in base_features:
        name = base_feature.name
        derivatives.append(base_feature)
        derivatives.append(sp.symbols(r"\dot{" + name + "}"))
        derivatives.append(sp.symbols(r"\ddot{" + name + "}"))

    subs = [(feature, 1) for feature in reversed(derivatives)]

    for part in eq.rhs.args:
        coeff = part.subs(subs)
        coefficients.append(coeff)

    return coefficients
