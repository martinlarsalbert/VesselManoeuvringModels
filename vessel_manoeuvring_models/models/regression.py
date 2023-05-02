import pandas as pd
import sympy as sp
from abc import ABC, abstractmethod

from vessel_manoeuvring_models.models.diff_eq_to_matrix import DiffEqToMatrix
from vessel_manoeuvring_models.symbols import *
from vessel_manoeuvring_models.parameters import *
import statsmodels.api as sm
from vessel_manoeuvring_models.visualization.regression import (
    show_pred_captive,
    show_pred,
    plot_pred,
    plot_pred_captive,
)
from vessel_manoeuvring_models.prime_system import PrimeSystem
from vessel_manoeuvring_models.models.vmm import ModelSimulator, FullModelSimulator
from vessel_manoeuvring_models.substitute_dynamic_symbols import lambdify, run
import pickle
import dill
from vessel_manoeuvring_models.models.vmm import Simulator
from scipy.linalg import block_diag
import statsmodels.api as sm


class Regression:
    """Regress forces and moments (either from Captive tests or motions)"""

    N_label = "mz"
    Y_label = "fy"
    X_label = "fx"

    N_fancy_label = r"$m_z$"
    Y_fancy_label = r"$f_y$"
    X_fancy_label = r"$f_x$"

    def __init__(
        self,
        vmm: ModelSimulator,
        data: pd.DataFrame,
        added_masses: dict,
        ship_parameters: dict,
        base_features=[delta, u, v, r, thrust],
        exclude_parameters: dict = {},
        connect_equations_Y_N_rudder=True,
        **kwargs,
    ):
        """[summary]

        Parameters
        ----------
        vmm : ModelSimulator
            vessel manoeuvring model
            either specified as:
            1) model simulator object
            2) or python module example: :func:`~vessel_manoeuvring_models.models.vmm_linear`
        data : pd.DataFrame
            Data to be regressed in SI units!
            That data should be a time series:
            index: time
            And the states:
            u,v,r,u1d,v1d,r1d
            and the inputs:
            delta,(thrust)

        added_masses : dict or pd.DataFrame with row: "prime"
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
        exclude_parameters : dict, optional
            Exclude some parameters from the regression by instead providing their value.
            Ex:
            exclude_parameters = {'Xthrust':0.95}
            means that Xthrust parameter will not be regressed, instead a value of 0.95 will be used.
        base_features : list, optional
            states and inputs to build features from defined in a list with sympy symbols, by default [delta, u, v, r]

        connect_equations_Y_N_rudder : True
        Should Ndelta = Ydelta*x_r be used to reduce the parameters?

        """

        self.ship_parameters = ship_parameters
        self.ps = PrimeSystem(L=ship_parameters["L"], rho=ship_parameters["rho"])
        self.ship_parameters_prime = self.ps.prime(self.ship_parameters)
        self.connected_parameters_Y = pd.Series(dtype=float)

        if isinstance(added_masses, pd.DataFrame):
            self.added_masses = added_masses["prime"]
        else:
            self.added_masses = added_masses

        self.data = data.copy()

        self.data["U"] = np.sqrt(self.data["u"] ** 2 + self.data["v"] ** 2)

        self.data_prime = self.ps.prime(self.data, U=self.data["U"])
        self.base_features = base_features
        self.connect_equations_Y_N_rudder = connect_equations_Y_N_rudder

        ## Simplify this:
        self.X_eq = vmm.X_eq
        self.Y_eq = vmm.Y_eq
        self.N_eq = vmm.N_eq

        self.X_qs_eq = vmm.X_qs_eq
        self.Y_qs_eq = vmm.Y_qs_eq
        self.N_qs_eq = vmm.N_qs_eq

        self.exclude_parameters = pd.Series(exclude_parameters)

        self.calculate_exclude_parameters_denominators()
        self.diff_equations()
        self.calculate_features_and_labels()
        self.model_N = self._fit_N()
        self.model_Y = self._fit_Y()
        self.model_X = self._fit_X()

        self.parameters = self.collect_parameters(vmm=vmm)

        # if isinstance(vmm, ModelSimulator):
        #    self.simulator = vmm
        # else:
        #    self.simulator = vmm.simulator
        self.simulator = Simulator(X_eq=self.X_eq, Y_eq=self.Y_eq, N_eq=self.N_eq)
        self.simulator.define_quasi_static_forces(
            X_qs_eq=self.X_qs_eq, Y_qs_eq=self.Y_qs_eq, N_qs_eq=self.N_qs_eq
        )

    def calculate_exclude_parameters_denominators(self):
        """excluded parameters are divided by these factors when they are excluded and moved to LHS.
        For force regression where both LHS and RHS are the same physical unit these parameters are 1.
        """
        self.exclude_parameters_denominator_N = 1
        self.exclude_parameters_denominator_Y = 1
        self.exclude_parameters_denominator_X = 1

    def diff_equations(self):

        N_ = sp.symbols("N_")
        self.diff_eq_N = DiffEqToMatrix(
            ode=self.N_qs_eq.subs(N_D, N_),
            label=N_,
            base_features=self.base_features,
            exclude_parameters=self.exclude_parameters,
            exclude_parameters_denominator=self.exclude_parameters_denominator_N,
        )

        Y_ = sp.symbols("Y_")
        self.diff_eq_Y = DiffEqToMatrix(
            ode=self.Y_qs_eq.subs(Y_D, Y_),
            label=Y_,
            base_features=self.base_features,
            exclude_parameters=self.exclude_parameters,
            exclude_parameters_denominator=self.exclude_parameters_denominator_Y,
        )

        X_ = sp.symbols("X_")
        ode = self.X_qs_eq.subs(
            [
                (X_D, X_),
            ]
        )

        self.diff_eq_X = DiffEqToMatrix(
            ode=ode,
            label=X_,
            base_features=self.base_features,
            exclude_parameters=self.exclude_parameters,
            exclude_parameters_denominator=self.exclude_parameters_denominator_X,
        )

    def results_summary_X(self):
        return results_summary_to_dataframe(self.model_X)

    def results_summary_Y(self):
        summary = results_summary_to_dataframe(self.model_Y)

        # Adding connected parameters:
        df = pd.DataFrame(columns=summary.columns)
        df["coeff"] = summary["coeff"].combine_first(self.connected_parameters_Y)
        summary = summary.combine_first(df)
        return summary

    def results_summary_N(self):
        return results_summary_to_dataframe(self.model_N)

    def collect_parameters(self, **kwargs):

        self.result_summaries = {
            "X": self.results_summary_X(),
            "Y": self.results_summary_Y(),
            "N": self.results_summary_N(),
        }

        return self._collect()

    def _collect(self, source="coeff"):

        df_parameters_all = pd.DataFrame()

        for dof, other in self.result_summaries.items():
            df_parameters_all = df_parameters_all.combine_first(other)

        df_parameters_all.rename(columns={source: "regressed"}, inplace=True)

        return df_parameters_all

    def calculate_features_and_labels(self):

        self.X_N, self.y_N = self.diff_eq_N.calculate_features_and_label(
            data=self.data_prime, y=self.data_prime[self.N_label]
        )

        self.X_Y, self.y_Y = self.diff_eq_Y.calculate_features_and_label(
            data=self.data_prime, y=self.data_prime[self.Y_label]
        )

        self.X_X, self.y_X = self.diff_eq_X.calculate_features_and_label(
            data=self.data_prime, y=self.data_prime[self.X_label]
        )

    def _fit_N(self):
        model_N = sm.OLS(self.y_N, self.X_N, hasconst=False)

        result = model_N.fit()

        if self.connect_equations_Y_N_rudder:
            # Feed regressed parameters as possible excludes to the Y-regression:

            connected_parameters = self.calculate_connected_parameters_N(result.params)
            self.diff_eq_Y.exclude_parameters = connected_parameters.combine_first(
                self.diff_eq_Y.exclude_parameters
            )

            self.diff_eq_Y.exclude_parameters = connected_parameters.combine_first(
                self.diff_eq_Y.exclude_parameters
            )

            ## Rerun to update (ugly)
            self.X_Y, self.y_Y = self.diff_eq_Y.calculate_features_and_label(
                data=self.data_prime, y=self.data_prime[self.Y_label]
            )

        return result

    def calculate_connected_parameters_N(self, params: pd.Series):

        if not "x_r" in self.ship_parameters_prime:
            return self.connected_parameters_Y

        delta_keys = [key for key in params.keys() if "delta" in key]
        for key in delta_keys:

            y_delta_key = f"Y{key[1:]}"

            self.connected_parameters_Y[y_delta_key] = (
                params[key] / self.ship_parameters_prime["x_r"]
            )

        return self.connected_parameters_Y

    def _fit_Y(self):
        model_Y = sm.OLS(self.y_Y, self.X_Y, hasconst=False)
        return model_Y.fit()

    def _fit_X(self):
        model_X = sm.OLS(self.y_X, self.X_X, hasconst=False)
        return model_X.fit()

    def predict(self, data: pd.DataFrame = None):

        df_predict = data.copy()

        df_predict["fx"] = self.predict_parameter_contributions_X(data=data).sum(axis=1)
        df_predict["fy"] = self.predict_parameter_contributions_Y(data=data).sum(axis=1)
        df_predict["mz"] = self.predict_parameter_contributions_N(data=data).sum(axis=1)

        return df_predict

    def _predict_parameter_contributions(
        self, diff_eq: DiffEqToMatrix, unit: str, data: pd.DataFrame = None
    ):

        if data is None:
            data_prime = self.data_prime
            data = self.data
        else:
            data_prime = self.ps.prime(data, U=data["U"])

        if data is None:
            data = self.data_prime

        X = diff_eq.calculate_features(data=data_prime)

        parameters_all = self.parameters["regressed"].copy()
        parameters_all = pd.concat((parameters_all, self.exclude_parameters))

        columns = list(set(parameters_all.index) & set(X.columns))
        parameters = parameters_all.loc[columns]

        force = X * parameters
        units = {key: unit for key in force.columns}

        return self.ps.df_unprime(force, units=units, U=data["U"])

    def predict_parameter_contributions_X(self, data: pd.DataFrame = None):
        return self._predict_parameter_contributions(
            diff_eq=self.diff_eq_X, unit="force", data=data
        )

    def predict_parameter_contributions_Y(self, data: pd.DataFrame = None):
        return self._predict_parameter_contributions(
            diff_eq=self.diff_eq_Y, unit="force", data=data
        )

    def predict_parameter_contributions_N(self, data: pd.DataFrame = None):
        return self._predict_parameter_contributions(
            diff_eq=self.diff_eq_N, unit="moment", data=data
        )

    def create_model(
        self,
        control_keys=["delta"],
        model_pos: sm.regression.linear_model.RegressionResultsWrapper = None,
        model_neg: sm.regression.linear_model.RegressionResultsWrapper = None,
        propeller_coefficients: dict = None,
        include_accelerations=False,
    ) -> ModelSimulator:
        """Create a ModelSimulator object from the regressed coefficients.
        There are however some things missing to obtain this:

        model_pos, model_neg and propeller_coefficients are provided this propeller model will be used.
        Otherwise thrust must be taken from the data that is resimulated.


        added masses are taken from: added_masses
        ship main dimensions and mass etc are taken from: ship_parameters

        Parameters
        ----------
        control_keys : list, optional
            [description], by default ["delta"]

        Returns
        -------
        ModelSimulator
            [description]
        """

        df_added_masses = pd.DataFrame(
            data=self.added_masses, index=["prime"]
        ).transpose()
        df_parameters_all = self.parameters.combine_first(df_added_masses)

        if "brix_lambda" in df_parameters_all:
            df_parameters_all.drop(columns=["brix_lambda"], inplace=True)

        df_parameters_all["regressed"] = df_parameters_all["regressed"].combine_first(
            df_parameters_all["prime"]
        )  # prefer regressed

        parameters = df_parameters_all["regressed"]
        parameters = pd.concat((parameters, self.exclude_parameters))

        if model_pos is None:

            return ModelSimulator(
                simulator=self.simulator,
                parameters=parameters,
                ship_parameters=self.ship_parameters,
                control_keys=control_keys,
                primed_parameters=True,
                prime_system=self.ps,
                include_accelerations=include_accelerations,
            )
        else:
            return FullModelSimulator(
                simulator=self.simulator,
                parameters=parameters,
                ship_parameters=self.ship_parameters,
                control_keys=control_keys,
                primed_parameters=True,
                prime_system=self.ps,
                model_pos=model_pos,
                model_neg=model_neg,
                propeller_coefficients=propeller_coefficients,
                include_accelerations=include_accelerations,
            )

    @staticmethod
    def show_pred(X, y, results, label):
        return show_pred_captive(X=X, y=y, results=results, label=label)

    @staticmethod
    def plot_pred(X, y, results, label):
        return plot_pred_captive(X=X, y=y, results=results, label=label)

    def show_pred_X(self):
        return self.show_pred(
            X=self.X_X, y=self.y_X, results=self.model_X, label=self.X_fancy_label
        )

    def show_pred_Y(self):
        return self.show_pred(
            X=self.X_Y, y=self.y_Y, results=self.model_Y, label=self.Y_fancy_label
        )

    def show_pred_N(self):
        return self.show_pred(
            X=self.X_N, y=self.y_N, results=self.model_N, label=self.N_fancy_label
        )

    def plot_pred_X(self):
        return self.plot_pred(
            X=self.X_X, y=self.y_X, results=self.model_X, label=self.X_fancy_label
        )

    def plot_pred_Y(self):
        return self.plot_pred(
            X=self.X_Y, y=self.y_Y, results=self.model_Y, label=self.Y_fancy_label
        )

    def plot_pred_N(self):
        return self.plot_pred(
            X=self.X_N, y=self.y_N, results=self.model_N, label=self.N_fancy_label
        )

    def show(self):
        self.show_pred_X()
        self.show_pred_Y()
        self.show_pred_N()

    def save(self, file_path: str):
        with open(file_path, mode="wb") as file:
            dill.dump(self, file)

    def __getstate__(self):
        def should_pickle(k):
            return not k in [
                "data",
                "data_prime",
            ]

        return {k: v for (k, v) in self.__dict__.items() if should_pickle(k)}


class ForceRegression(Regression):
    """Regressing a model from forces and moments, similar to captive tests or PMM tests."""

    N_label = "mz"
    Y_label = "fy"
    X_label = "fx"

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

    def plot_pred_X(self):
        return plot_pred_captive(
            X=self.X_X, y=self.y_X, results=self.model_X, label=r"$X$"
        )

    def plot_pred_Y(self):
        return plot_pred_captive(
            X=self.X_Y, y=self.y_Y, results=self.model_Y, label=r"$Y$"
        )

    def plot_pred_N(self):
        return plot_pred_captive(
            X=self.X_N, y=self.y_N, results=self.model_N, label=r"$N$"
        )


class MotionRegression(Regression):
    """Regressing a model from ship motions."""

    N_label = "r1d"
    Y_label = "v1d"
    X_label = "u1d"

    N_fancy_label = r"$\dot{r}$"
    Y_fancy_label = r"$\dot{v}$"
    X_fancy_label = r"$\dot{u}$"

    def collect_parameters(self, vmm):
        self.parameters = super().collect_parameters()
        self.decoupling(vmm=vmm)
        self.parameters = self._collect(source="decoupled")

        self.std = self.std[self.parameters.index].copy()
        self.covs = self.covs.loc[self.parameters.index, self.parameters.index].copy()

        return self.parameters

    def decoupling(self, vmm):

        # Remove exclude parameters:
        subs = [(p[key], 0) for key in self.exclude_parameters.keys() if key in p]
        X_eq = vmm.X_eq.subs(subs)
        Y_eq = vmm.Y_eq.subs(subs)
        N_eq = vmm.N_eq.subs(subs)

        A, b = sp.linear_eq_to_matrix([X_eq, Y_eq, N_eq], [u1d, v1d, r1d])

        subs = {value: key for key, value in p.items()}

        A_ = A * sp.matrices.MutableDenseMatrix([A_coeff, B_coeff, C_coeff])
        A_lambda = lambdify(A_.subs(subs))

        A_coeff_ = self.result_summaries["X"]["coeff"]
        B_coeff_ = self.result_summaries["Y"]["coeff"]
        C_coeff_ = self.result_summaries["N"]["coeff"]

        ship_parameters_prime = self.ps.prime(self.ship_parameters)

        # B_coeff_ = B_coeff_.append(self.connected_parameters_Y)
        # B_coeff_.sort_index(inplace=True)

        coeffs = run(
            A_lambda,
            A_coeff=A_coeff_.values,
            B_coeff=B_coeff_.values,
            C_coeff=C_coeff_.values,
            **self.parameters["regressed"],
            **self.added_masses,
            **ship_parameters_prime,
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

        B_coeff_bse = self.model_Y.bse
        B_coeff_bse = pd.concat((B_coeff_bse, 
            pd.Series({key: 0 for key in self.connected_parameters_Y.keys()})
        ))
        B_coeff_bse.sort_index(inplace=True)

        stds = run(
            A_lambda,
            A_coeff=self.model_X.bse.values,
            B_coeff=B_coeff_bse.values,
            C_coeff=self.model_N.bse.values,
            **self.parameters,
            **self.added_masses,
            **ship_parameters_prime,
        )

        self.std = pd.Series()
        self.std = pd.concat((self.std, pd.Series(stds[0][0], index=self.model_X.bse.index)))
        self.std = pd.concat((self.std, pd.Series(stds[1][0], index=B_coeff_bse.index)))
        self.std = pd.concat((self.std, pd.Series(stds[2][0], index=self.model_N.bse.index)))

        covs = run(
            A_lambda,
            A_coeff=self.model_X.cov_HC0,
            B_coeff=self.model_Y.cov_HC0,
            C_coeff=self.model_N.cov_HC0,
            **self.parameters,
            **self.added_masses,
            **ship_parameters_prime,
        )
        columns = (
            list(self.model_X.params.keys())
            + list(self.model_Y.params.keys())
            + list(self.model_N.params.keys())
        )
        self.covs = pd.DataFrame(
            block_diag(*covs[:, 0]), index=columns, columns=columns
        )

    def calculate_exclude_parameters_denominators(self):
        """excluded parameters are divided by these factors when they are excluded and moved to LHS.
        For motion regression the equations are divided by the mass inertia (including added mass)
        """
        ship_parameters_prime = self.ps.prime(self.ship_parameters)

        self.exclude_parameters_denominator_N = (
            ship_parameters_prime["I_z"]
            - self.added_masses["Nrdot"]  # (A bit unsure about this one...)
        )
        self.exclude_parameters_denominator_Y = (
            ship_parameters_prime["m"]
            - self.added_masses["Yvdot"]  # (A bit unsure about this one...)
        )

        self.exclude_parameters_denominator_X = (
            ship_parameters_prime["m"] - self.added_masses["Xudot"]
        )

    @staticmethod
    def show_pred(X, y, results, label):
        return show_pred(X=X, y=y, results=results, label=label)

    @staticmethod
    def plot_pred(X, y, results, label):
        return plot_pred(X=X, y=y, results=results, label=label)


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
