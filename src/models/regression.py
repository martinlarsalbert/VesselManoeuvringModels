from random import vonmisesvariate
import pandas as pd
import numpy as np
import sympy as sp
from abc import ABC, abstractmethod
from src import symbols

from src.parameters import Xudot_, df_parameters
from src.models.diff_eq_to_matrix import DiffEqToMatrix
from src.symbols import *
from src.parameters import *
import statsmodels.api as sm
from src.visualization.regression import show_pred_captive, show_pred
from src.prime_system import PrimeSystem
from src.models.vmm import ModelSimulator
from src.substitute_dynamic_symbols import lambdify, run


class Regression(ABC):
    """Base class for all regressions
    (intended to be inherited and extended)
    """

    def __init__(
        self, vmm: ModelSimulator, data: pd.DataFrame, base_features=[delta, u, v, r]
    ):
        """Regression

        Parameters
        ----------
        vmm : ModelSimulator
            vessel manoeuvring model
            either specified as:
            1) model simulator object
            2) or python module example: :func:`~src.models.vmm_linear`
        data : pd.DataFrame
            Data to be regressed
            Must contain the forces: fx,fy,mz
            and the inputs: delta, (thrust)
        base_features : list, optional
            states and inputs to build features from defined in a list with sympy symbols, by default [delta, u, v, r]
        """
        self.vmm = vmm
        self.data = data
        self.base_features = base_features

        self.diff_equations()
        self.model_N = self._fit_N()
        self.model_Y = self._fit_Y()
        self.model_X = self._fit_X()

        self.parameters = self.collect_parameters()

    def diff_equations(self):

        N_ = sp.symbols("N_")
        self.diff_eq_N = DiffEqToMatrix(
            ode=self.vmm.N_qs_eq.subs(N_D, N_),
            label=N_,
            base_features=self.base_features,
        )

        Y_ = sp.symbols("Y_")
        self.diff_eq_Y = DiffEqToMatrix(
            ode=self.vmm.Y_qs_eq.subs(Y_D, Y_),
            label=Y_,
            base_features=self.base_features,
        )

        X_ = sp.symbols("X_")
        ode = self.vmm.X_qs_eq.subs(
            [
                (X_D, X_),
            ]
        )

        self.diff_eq_X = DiffEqToMatrix(
            ode=ode, label=X_, base_features=self.base_features
        )

    def results_summary_X(self):
        return results_summary_to_dataframe(self.model_X)

    def results_summary_Y(self):
        return results_summary_to_dataframe(self.model_Y)

    def results_summary_N(self):
        return results_summary_to_dataframe(self.model_N)

    def collect_parameters(self):

        self.result_summaries = {
            "X": self.results_summary_X(),
            "Y": self.results_summary_Y(),
            "N": self.results_summary_N(),
        }

        return self._collect()

    def _collect(self, source="coeff"):

        df_parameters_all = pd.DataFrame()

        for other in self.result_summaries.values():
            df_parameters_all = df_parameters_all.combine_first(other)

        df_parameters_all.rename(columns={source: "regressed"}, inplace=True)

        return df_parameters_all

    def create_model(
        self,
        added_masses: dict,
        ship_parameters: dict,
        ps: PrimeSystem,
        control_keys=["delta"],
    ) -> ModelSimulator:
        """Create a ModelSimulator object from the regressed coefficients.
        There are however some things missing to obtain this:

        added masses are taken from: added_masses
        ship main dimensions and mass etc are taken from: ship_parameters

        Parameters
        ----------
        added_masses : dict
            Added masses are taken from here

        ship_parameters : dict
            ship main dimensions and mass etc

        ps : PrimeSystem
            [description]
        control_keys : list, optional
            [description], by default ["delta"]

        Returns
        -------
        ModelSimulator
            [description]
        """

        df_parameters_all = self.parameters.combine_first(added_masses)

        if "brix_lambda" in df_parameters_all:
            df_parameters_all.drop(columns=["brix_lambda"], inplace=True)

        df_parameters_all["regressed"] = df_parameters_all["regressed"].combine_first(
            df_parameters_all["prime"]
        )  # prefer regressed

        if isinstance(self.vmm, ModelSimulator):
            simulator = self.vmm
        else:
            simulator = self.vmm.simulator

        return ModelSimulator(
            simulator=simulator,
            parameters=df_parameters_all["regressed"],
            ship_parameters=ship_parameters,
            control_keys=control_keys,
            primed_parameters=True,
            prime_system=ps,
        )

    @property
    def X_N(self):
        X = self.diff_eq_N.calculate_features(data=self.data)
        return X

    @property
    @abstractmethod
    def y_N(self):
        y = self.diff_eq_N.calculate_label(y=self.data["mz"])
        return y

    @property
    def X_Y(self):
        X = self.diff_eq_Y.calculate_features(data=self.data)
        return X

    @property
    @abstractmethod
    def y_Y(self):
        y = self.diff_eq_Y.calculate_label(y=self.data["fy"])
        return y

    @property
    def X_X(self):
        X = self.diff_eq_X.calculate_features(data=self.data)
        return X

    @property
    @abstractmethod
    def y_X(self):
        y = self.diff_eq_X.calculate_label(y=self.data["fx"])
        return y

    def _fit_N(self):
        model_N = sm.OLS(self.y_N, self.X_N)
        return model_N.fit()

    def _fit_Y(self):
        model_Y = sm.OLS(self.y_Y, self.X_Y)
        return model_Y.fit()

    def _fit_X(self):
        model_X = sm.OLS(self.y_X, self.X_X)
        return model_X.fit()

    @abstractmethod
    def show_pred_X(self):
        return show_pred_captive(
            X=self.X_X, y=self.y_X, results=self.model_X, label=r"$X$"
        )

    @abstractmethod
    def show_pred_Y(self):
        return show_pred_captive(
            X=self.X_Y, y=self.y_Y, results=self.model_Y, label=r"$Y$"
        )

    @abstractmethod
    def show_pred_N(self):
        return show_pred_captive(
            X=self.X_N, y=self.y_N, results=self.model_N, label=r"$N$"
        )

    def show(self):
        self.show_pred_X()
        self.show_pred_Y()
        self.show_pred_N()


class ForceRegression(Regression):
    """Regressing a model from forces and moments, similar to captive tests or PMM tests."""

    @property
    def y_N(self):
        y = self.diff_eq_N.calculate_label(y=self.data["mz"])
        return y

    @property
    def y_Y(self):
        y = self.diff_eq_Y.calculate_label(y=self.data["fy"])
        return y

    @property
    def y_X(self):
        y = self.diff_eq_X.calculate_label(y=self.data["fx"])
        return y

    def show_pred_X(self):
        return show_pred_captive(
            X=self.X_X, y=self.y_X, results=self.model_X, label=r"$X$"
        )

    def show_pred_Y(self):
        return show_pred_captive(
            X=self.X_Y, y=self.y_Y, results=self.model_Y, label=r"$Y$"
        )

    def show_pred_N(self):
        return show_pred_captive(
            X=self.X_N, y=self.y_N, results=self.model_N, label=r"$N$"
        )

    def show_pred_X(self):
        return show_pred_captive(
            X=self.X_X, y=self.y_X, results=self.model_X, label=r"$X$"
        )

    def show_pred_Y(self):
        return show_pred_captive(
            X=self.X_Y, y=self.y_Y, results=self.model_Y, label=r"$Y$"
        )

    def show_pred_N(self):
        return show_pred_captive(
            X=self.X_N, y=self.y_N, results=self.model_N, label=r"$N$"
        )


class MotionRegression(Regression):
    """Regressing a model from ship motions."""

    def __init__(
        self,
        vmm: ModelSimulator,
        data: pd.DataFrame,
        added_masses: dict,
        ship_parameters: dict,
        prime_system: PrimeSystem,
        base_features=[delta, u, v, r],
    ):
        """[summary]

        Parameters
        ----------
        vmm : ModelSimulator
            vessel manoeuvring model
            either specified as:
            1) model simulator object
            2) or python module example: :func:`~src.models.vmm_linear`
        data : pd.DataFrame
            Data to be regressed.
            That data should be a time series:
            index: time
            And the states:
            u,v,r,u1d,v1d,r1d
            and the inputs:
            delta,(thrust)

        added_masses : dict
            added masses in prime system units
        ship_parameters : dict
            ship parameters in SI units,
            ex:
            {
                "L": 100,        # Ship length [m]
                "rho": 1025,     # water density [kg/m3]
                "I_z": 100000000,# yaw mass moment of inertia [kg*m**2]
                "m": 10000000,   # mass of ship [kg]
                "x_G": 2.5,     # Longitudinal position of CG rel lpp/2 [m]
            }
        prime_system : PrimeSystem
            prime system object for the current ship
        base_features : list, optional
            states and inputs to build features from defined in a list with sympy symbols, by default [delta, u, v, r]
        """

        self.ship_parameters = ship_parameters
        self.ps = prime_system
        self.added_masses = added_masses

        super().__init__(vmm=vmm, data=data, base_features=base_features)

    def collect_parameters(self):
        self.parameters = super().collect_parameters()
        self.decoupling()
        self.parameters = self._collect(source="decoupled")
        return self.parameters

    def decoupling(self):

        A, b = sp.linear_eq_to_matrix(
            [self.vmm.X_eq, self.vmm.Y_eq, self.vmm.N_eq], [u1d, v1d, r1d]
        )

        subs = {value: key for key, value in p.items()}

        A_ = A * sp.matrices.MutableDenseMatrix([A_coeff, B_coeff, C_coeff])
        A_lambda = lambdify(A_.subs(subs))

        A_coeff_ = self.result_summaries["X"]["coeff"]
        B_coeff_ = self.result_summaries["Y"]["coeff"]
        C_coeff_ = self.result_summaries["N"]["coeff"]

        ship_parameters_prime = self.ps.prime(self.ship_parameters)

        coeffs = run(
            A_lambda,
            A_coeff=A_coeff_.values,
            B_coeff=B_coeff_.values,
            C_coeff=C_coeff_.values,
            **self.parameters["regressed"],
            **self.added_masses["prime"],
            **ship_parameters_prime
        )

        self.result_summaries["X"]["decoupled"] = coeffs[0][0]
        self.result_summaries["Y"]["decoupled"] = coeffs[1][0]
        self.result_summaries["N"]["decoupled"] = coeffs[2][0]

        ## Removing the centripetal and coriolis forces:
        x_G_ = ship_parameters_prime["x_G"]
        m_ = ship_parameters_prime["m"]

        if "Xrr" in self.result_summaries["X"].index:
            self.result_summaries["X"].loc["Xrr", "decoupled"] += -m_ * x_G_

        if "Xvr" in self.result_summaries["X"].index:
            self.result_summaries["X"].loc["Xvr", "decoupled"] += -m_

        if "Yur" in self.result_summaries["Y"].index:
            self.result_summaries["Y"].loc["Yur", "decoupled"] += m_

        if "Nur" in self.result_summaries["N"].index:
            self.result_summaries["N"].loc["Nur", "decoupled"] += m_ * x_G_

    @property
    def y_N(self):
        y = self.diff_eq_N.calculate_label(y=self.data["r1d"])
        return y

    @property
    def X_Y(self):
        X = self.diff_eq_Y.calculate_features(data=self.data)
        return X

    @property
    def y_Y(self):
        y = self.diff_eq_Y.calculate_label(y=self.data["v1d"])
        return y

    @property
    def X_X(self):
        X = self.diff_eq_X.calculate_features(data=self.data)
        return X

    @property
    def y_X(self):
        y = self.diff_eq_X.calculate_label(y=self.data["u1d"])
        return y

    def show_pred_X(self):
        return show_pred(
            X=self.X_X, y=self.y_X, results=self.model_X, label=r"$\dot{u}$"
        )

    def show_pred_Y(self):
        return show_pred(
            X=self.X_Y, y=self.y_Y, results=self.model_Y, label=r"$\dot{v}$"
        )

    def show_pred_N(self):
        return show_pred(
            X=self.X_N, y=self.y_N, results=self.model_N, label=r"$\dot{r}$"
        )


def results_summary_to_dataframe(results):
    """take the result of an statsmodel results
    table and transforms it into a dataframe"""
    pvals = results.pvalues
    coeff = results.params
    conf_lower = results.conf_int()[0]
    conf_higher = results.conf_int()[1]

    results_df = pd.DataFrame(
        {
            "$P_{value}$": pvals,
            "coeff": coeff,
            "$conf_{lower}$": conf_lower,
            "$conf_{higher}$": conf_higher,
        }
    )

    # Reordering...
    results_df = results_df[
        ["coeff", "$P_{value}$", "$conf_{lower}$", "$conf_{higher}$"]
    ]
    return results_df
