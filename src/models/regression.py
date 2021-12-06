from random import vonmisesvariate
import pandas as pd
import numpy as np
import sympy as sp
from src.substitute_dynamic_symbols import lambdify, run
from src import symbols

from src.parameters import Xudot_, df_parameters
from src.models.diff_eq_to_matrix import DiffEqToMatrix
from src.symbols import *
import statsmodels.api as sm
from src.visualization.regression import show_pred_captive
from src.prime_system import PrimeSystem
from src.models.vmm import ModelSimulator


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


class ForceRegression:
    def __init__(self, vmm, data: pd.DataFrame, base_features=[delta, u, v, r]):

        self.vmm = vmm
        self.data = data
        self.base_features = base_features

        self.diff_equations()
        self.model_N = self._fit_N()
        self.model_Y = self._fit_Y()
        self.model_X = self._fit_X()

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

    @property
    def X_N(self):
        X = self.diff_eq_N.calculate_features(data=self.data)
        return X

    @property
    def y_N(self):
        y = self.diff_eq_N.calculate_label(y=self.data["mz"])
        return y

    @property
    def X_Y(self):
        X = self.diff_eq_Y.calculate_features(data=self.data)
        return X

    @property
    def y_Y(self):
        y = self.diff_eq_Y.calculate_label(y=self.data["fy"])
        return y

    @property
    def X_X(self):
        X = self.diff_eq_X.calculate_features(data=self.data)
        return X

    @property
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

    def results_summary_X(self):
        return results_summary_to_dataframe(self.model_X)

    def results_summary_Y(self):
        return results_summary_to_dataframe(self.model_Y)

    def results_summary_N(self):
        return results_summary_to_dataframe(self.model_N)

    def parameters(self):
        df_parameters_all = pd.DataFrame()

        for other in [
            self.results_summary_X(),
            self.results_summary_Y(),
            self.results_summary_N(),
        ]:
            df_parameters_all = df_parameters_all.combine_first(other)

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

        df_parameters_all = self.parameters().combine_first(added_masses)

        df_parameters_all.rename(columns={"coeff": "regressed"}, inplace=True)
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
