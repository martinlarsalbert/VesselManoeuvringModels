import os.path
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vessel_manoeuvring_models.visualization.plot import track_plot, plot
from vessel_manoeuvring_models.substitute_dynamic_symbols import run, lambdify
from IPython.display import display
from os import stat
from sklearn.metrics import r2_score
from vessel_manoeuvring_models.models.diff_eq_to_matrix import DiffEqToMatrix
import sympy as sp
from vessel_manoeuvring_models.symbols import *
import matplotlib.ticker as plticker
from sklearn.metrics import r2_score
from vessel_manoeuvring_models.apparent_wind import (
    true_wind_speed_to_apparent,
    true_wind_angle_to_apparent,
)
from scipy.spatial.transform import Rotation as R
from vessel_manoeuvring_models.angles import smallest_signed_angle


class Result:
    def __init__(
        self,
        simulator,
        solution,
        df_model_test,
        df_control,
        ship_parameters,
        parameters,
        y0,
        include_accelerations=True,
        name="simulation",
    ):

        self.simulator = simulator
        self.solution = solution
        self.df_model_test = df_model_test
        self.df_control = df_control
        self.ship_parameters = ship_parameters
        self.parameters = parameters
        self.y0 = y0
        self.include_accelerations = include_accelerations
        self.name = name

    @property
    def simulation_result(self):

        columns = list(self.y0.keys())
        df_result = pd.DataFrame(
            data=self.solution.y.T, columns=columns, index=self.solution.t
        )

        for key in self.df_control:
            df_result[key] = self.df_control[key].iloc[0 : len(df_result)].values

        try:
            df_result["beta"] = -np.arctan2(df_result["v"], df_result["u"])
        except:
            pass

        try:
            df_result["U"] = np.sqrt(df_result["u"] ** 2 + df_result["v"] ** 2)
        except:
            pass

        if "tws" in self.df_control and "twa" in self.df_control:
            x01d = derivative(df_result, key="x0")
            y01d = derivative(df_result, key="y0")
            df_result["cog"] = np.arctan2(y01d, x01d)
            df_result["aws"] = true_wind_speed_to_apparent(**df_result)
            awa = true_wind_angle_to_apparent(**df_result)
            df_result["awa"] = smallest_signed_angle(
                awa - (awa.iloc[0] - smallest_signed_angle(awa.iloc[0]))
            )

        return df_result

    @property
    def result(self):
        df_result = self.simulation_result

        if self.include_accelerations:
            df_result = pd.concat([df_result, self.accelerations], axis=1)

        return df_result

    @property
    def X_qs(self) -> pd.Series:
        """Hydrodynamic force from ship in X-direction during simulation"""
        return self._calcualte_qs_force(
            function=self.simulator.X_qs_lambda, unit="force"
        )

    @property
    def Y_qs(self) -> pd.Series:
        """Hydrodynamic force from ship in Y-direction during simulation"""
        return self._calcualte_qs_force(
            function=self.simulator.Y_qs_lambda, unit="force"
        )

    @property
    def N_qs(self) -> pd.Series:
        """Hydrodynamic force from ship in N-direction during simulation"""
        return self._calcualte_qs_force(
            function=self.simulator.N_qs_lambda, unit="moment"
        )

    def _calcualte_qs_force(self, function, unit):
        df_result = self.simulation_result.copy()

        if self.simulator.primed_parameters:
            df_result_prime = self.simulator.prime_system.prime(
                df_result, U=df_result["U"]
            )
            X_qs_ = run(function=function, **df_result_prime, **self.parameters)
            return self.simulator.prime_system._unprime(
                X_qs_, unit=unit, U=df_result["U"]
            )
        else:
            return run(function=function, **df_result, **self.parameters)

    @property
    def accelerations(self):
        df_result = self.simulation_result.copy()

        if self.simulator.primed_parameters:
            df_result_prime = self.simulator.prime_system.prime(
                df_result, U=df_result["U"]
            )

            inputs = df_result_prime
            inputs["U0"] = inputs.iloc[0]["U"]

            u1d_prime, v1d_prime, r1d_prime = run(
                function=self.simulator.acceleration_lambda,
                X_qs=run(
                    function=self.simulator.X_qs_lambda,
                    **inputs,
                    **self.parameters,
                    **self.ship_parameters,
                ),
                Y_qs=run(
                    function=self.simulator.Y_qs_lambda,
                    **inputs,
                    **self.parameters,
                    **self.ship_parameters,
                ),
                N_qs=run(
                    function=self.simulator.N_qs_lambda,
                    **inputs,
                    **self.parameters,
                    **self.ship_parameters,
                ),
                **inputs,
                **self.parameters,
                **self.simulator.ship_parameters_prime,
            )

            df_accelerations_prime = pd.DataFrame(index=df_result.index)
            df_accelerations_prime["u1d"] = u1d_prime[0]
            df_accelerations_prime["v1d"] = v1d_prime[0]
            df_accelerations_prime["r1d"] = r1d_prime[0]
            df_accelerations = self.simulator.prime_system.unprime(
                df_accelerations_prime, U=df_result["U"]
            )
        else:

            inputs = df_result
            inputs["U0"] = inputs.iloc[0]["U"]

            u1d, v1d, r1d = run(
                function=self.simulator.acceleration_lambda,
                X_qs=run(
                    function=self.simulator.X_qs_lambda,
                    inputs=inputs,
                    **self.parameters,
                ),
                Y_qs=run(
                    function=self.simulator.Y_qs_lambda,
                    inputs=inputs,
                    **self.parameters,
                ),
                N_qs=run(
                    function=self.simulator.N_qs_lambda,
                    inputs=inputs,
                    **self.parameters,
                ),
                inputs=inputs,
                **self.parameters,
                **self.ship_parameters,
            )

            df_accelerations = pd.DataFrame(index=df_result.index)
            df_accelerations["u1d"] = u1d[0]
            df_accelerations["v1d"] = v1d[0]
            df_accelerations["r1d"] = r1d[0]

        return df_accelerations

    def plot_compare(self, compare=True):

        self.track_plot(compare=compare)
        self.plot(compare=compare)

    def track_plot(self, ax=None, compare=True):
        if ax is None:
            fig, ax = plt.subplots()

        track_plot(
            df=self.simulation_result,
            lpp=self.ship_parameters["L"],
            beam=self.ship_parameters["B"],
            ax=ax,
            label=self.name,
            color="green",
        )

        if compare:
            track_plot(
                df=self.df_model_test,
                lpp=self.ship_parameters["L"],
                beam=self.ship_parameters["B"],
                ax=ax,
                label="data",
            )
            ax.legend()
        return ax

    def plot(self, subplot=True, compare=True):

        if compare:

            dataframes = {
                self.name: self.simulation_result,
                "data": self.df_model_test,
            }

        else:
            dataframes = {
                self.name: self.simulation_result,
            }

        return plot(dataframes=dataframes)

    def plot_zigzag(self, ax=None, compare=True):

        if ax is None:
            fig, ax = plt.subplots()

        df_result = self.simulation_result.copy()
        df_result["psi_deg"] = np.rad2deg(df_result["psi"])
        df_result["-delta_deg"] = -np.rad2deg(df_result["delta"])
        df_result.plot(y=["psi_deg", "-delta_deg"], ax=ax)

        if compare:
            df_result2 = self.df_model_test.copy()
            df_result2["psi_deg"] = np.rad2deg(df_result2["psi"])
            df_result2["-delta_deg"] = -np.rad2deg(df_result2["delta"])
            df_result2.plot(y=["psi_deg"], style="--", ax=ax)

        loc = plticker.MultipleLocator(
            base=1.0
        )  # this locator puts ticks at regular intervals
        ax.yaxis.set_major_locator(loc)
        ax.grid()

    def save(self, path: str):
        """Save the simulation to a csv file"""
        self.result.to_csv(path, index=True)

    def to_mlflow(self, artifact_dir="artifacts"):
        """log this run to mlflow

        Ex:
        This method is intended to be within a mlflow.start_run with statement:

        mlflow.set_experiment(run_params['experiment'])
        with mlflow.start_run(run_name='test') as run:

            log_params = run_params.copy()
            log_params.pop('experiment')
            mlflow.log_params(run_params)

            -->result.to_mlflow()

        Parameters
        ----------
        artifact_dir : str, optional
            [description], by default 'artifacts'
        """

        import mlflow

        if not os.path.exists(artifact_dir):
            os.mkdir(artifact_dir)

        save_path = os.path.join(artifact_dir, "result.csv")
        self.save(path=save_path)
        mlflow.log_artifact(save_path)

        fig, ax = plt.subplots()
        fig.set_size_inches(15, 10)
        self.track_plot(compare=True, ax=ax)
        mlflow.log_figure(fig, "track_plot.png")

        fig = self.plot(compare=True)
        fig.set_size_inches(15, 10)
        plt.tight_layout()
        mlflow.log_figure(fig, "signals.png")

        ## R2 score
        interesting = list(self.simulation_result.keys())
        for key in self.df_control.keys():
            interesting.remove(key)

        r2s = {
            f"r2_{key}": r2_score(
                y_true=self.df_model_test[key], y_pred=self.result[key]
            )
            for key in interesting
        }
        r2s = pd.Series(r2s)
        r2s["r2"] = r2s.mean()  # Mean r2
        mlflow.log_metrics(r2s)

        self.plot_parameter_contributions()

    def simulate_parameter_contributions(self):

        model = self.simulator
        df_result_prime = model.prime_system.prime(self.result, U=self.result["U"])

        base_features = [delta, u, v, r, aws, awa, thrust, A_XV, A_YV, rho_A, L]
        X_ = sp.symbols("X_")
        diff_eq_X = DiffEqToMatrix(
            ode=model.X_qs_eq.subs(X_D, X_),
            label=X_,
            base_features=base_features,
        )
        parameters = model.prime_system.prime(self.ship_parameters)
        X = diff_eq_X.calculate_features(
            data=df_result_prime,
            parameters=parameters,
        )
        X_parameters = self.simulator.parameters[
            model.get_coefficients_X(sympy_symbols=False)
        ]
        X_forces = X * X_parameters
        X_forces.index = df_result_prime.index

        Y_ = sp.symbols("Y_")
        diff_eq_Y = DiffEqToMatrix(
            ode=model.Y_qs_eq.subs(Y_D, Y_), label=Y_, base_features=base_features
        )
        X = diff_eq_Y.calculate_features(
            data=df_result_prime,
            parameters=parameters,
        )
        Y_parameters = model.parameters[model.get_coefficients_Y(sympy_symbols=False)]
        Y_forces = X * Y_parameters
        Y_forces.index = df_result_prime.index

        N_ = sp.symbols("N_")
        diff_eq_N = DiffEqToMatrix(
            ode=model.N_qs_eq.subs(N_D, N_), label=N_, base_features=base_features
        )
        X = diff_eq_N.calculate_features(data=df_result_prime, parameters=parameters)
        N_parameters = model.parameters[model.get_coefficients_N(sympy_symbols=False)]
        N_forces = X * N_parameters
        N_forces.index = df_result_prime.index

        return X_forces, Y_forces, N_forces

    def plot_parameter_contributions(self, to_mlflow=True):

        X_forces, Y_forces, N_forces = self.simulate_parameter_contributions()

        fig_X = px.line(X_forces, y=X_forces.columns, width=800, height=350, title="X")
        display(fig_X)

        fig_Y = px.line(Y_forces, y=Y_forces.columns, width=800, height=350, title="Y")
        display(fig_Y)

        fig_N = px.line(N_forces, y=N_forces.columns, width=800, height=350, title="N")
        display(fig_N)

        if to_mlflow:
            mlflow.log_figure(fig_X, "parameter_contributions_X.html")
            mlflow.log_figure(fig_Y, "parameter_contributions_Y.html")
            mlflow.log_figure(fig_N, "parameter_contributions_N.html")

    def score(self) -> pd.Series:
        """R2 score

        Returns
        -------
        pd.Series
            r2 score for each signal
        """

        r2s = pd.Series(dtype=float)

        for key in self.result.columns:

            if key in self.df_control:
                continue

            if key in self.df_model_test:

                r2s[key] = r2_score(
                    y_true=self.df_model_test[key], y_pred=self.result[key]
                )

        return r2s


def derivative(df, key):
    d = np.diff(df[key]) / np.diff(df.index)
    d = np.concatenate((d, [d[-1]]))
    return d
