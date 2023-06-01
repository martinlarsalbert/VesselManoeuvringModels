"""Module for Vessel Manoeuvring Model (VMM) simulation

References:
[1] Matusiak, Jerzy. Dynamics of a Rigid Ship - with Applications, 3rd Edition, 2021.


"""

import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame

from scipy.integrate import solve_ivp
import sympy as sp
from vessel_manoeuvring_models.symbols import *
import warnings

from vessel_manoeuvring_models.parameters import df_parameters
from vessel_manoeuvring_models.prime_system import PrimeSystem
import statsmodels.api as sm

from vessel_manoeuvring_models.symbols import *

p = df_parameters["symbol"]


from vessel_manoeuvring_models.substitute_dynamic_symbols import run, lambdify
from scipy.spatial.transform import Rotation as R
import vessel_manoeuvring_models.prime_system

import vessel_manoeuvring_models.models.diff_eq_to_matrix
import dill
from vessel_manoeuvring_models.models.result import Result
from sklearn.utils import Bunch
from copy import deepcopy
from sympy.printing import pretty
from vessel_manoeuvring_models.models.propeller import predictor
from vessel_manoeuvring_models.apparent_wind import (
    true_wind_speed_to_apparent,
    true_wind_angle_to_apparent,
)
from vessel_manoeuvring_models.angles import smallest_signed_angle


class VMM:
    """Vessel Manoeuvring Model
    Holding the equation of motions (EOM) and damping force equation for one model
    """

    def __init__(self, X_eq: sp.Eq, Y_eq: sp.Eq, N_eq: sp.Eq):
        """[summary]

        Parameters
        ----------
        X_eq : sp.Eq
            Equation in X-direction
        Y_eq : sp.Eq
            Equation in Y-direction
        N_eq : sp.Eq
            Equation in N-direction

        Example:

        X_eq:
           /           2          \                                             2
        m*\\dot{u} - r *x_G - r*v/ = X_{\dot{u}}*\dot{u} + X_{deltadelta}*delta  + X_{

             2                               2
        rr}*r  + X_{thrust}*thrust + X_{uu}*u  + X_{u}*u + X_{vdelta}*delta*v + X_{vr}


        *r*v

        Y_eq:

        m*(\dot{r}*x_G + \dot{v} + r*u) = Y_{\dot{r}}*\dot{r} + Y_{\dot{v}}*\dot{v} +

                                               2
        Y_{delta}*delta + Y_{rdeltadelta}*delta *r + Y_{r}*r + Y_{ur}*r*u + Y_{uv}*u*v

                                          2
         + Y_{u}*u + Y_{vdeltadelta}*delta *v + Y_{v}*v

        N_eq:
        I_z*\dot{r} + m*x_G*(\dot{v} + r*u) = N_{\dot{r}}*\dot{r} + N_{\dot{v}}*\dot{v

                                                   2
        } + N_{delta}*delta + N_{rdeltadelta}*delta *r + N_{r}*r + N_{ur}*r*u + N_{uv}

                                              2
        *u*v + N_{u}*u + N_{vdeltadelta}*delta *v + N_{v}*v

        """

        self.X_eq = X_eq
        self.Y_eq = Y_eq
        self.N_eq = N_eq

        self.X_qs_eq = sp.Eq(
            X_D, self.X_eq.subs([(m, 0), (I_z, 0), (u1d, 0), (v1d, 0), (r1d, 0)]).rhs
        )
        self.Y_qs_eq = sp.Eq(
            Y_D, self.Y_eq.subs([(m, 0), (I_z, 0), (u1d, 0), (v1d, 0), (r1d, 0)]).rhs
        )
        self.N_qs_eq = sp.Eq(
            N_D, self.N_eq.subs([(m, 0), (I_z, 0), (u1d, 0), (v1d, 0), (r1d, 0)]).rhs
        )

        self.X_eq_separated = self.separated_equation(
            eq=self.X_eq, eq_qs=self.X_qs_eq, damping_function=X_D
        )
        self.Y_eq_separated = self.separated_equation(
            eq=self.Y_eq, eq_qs=self.Y_qs_eq, damping_function=Y_D
        )
        self.N_eq_separated = self.separated_equation(
            eq=self.N_eq, eq_qs=self.N_qs_eq, damping_function=N_D
        )

    @staticmethod
    def separated_equation(eq, eq_qs, damping_function):
        """Separating the equation into the damping_function (=X_D, Y_D, N_D) and inertia forces."""

        states = [u, v, r, delta, thrust]
        parameters = eq_qs.rhs.free_symbols - set(states)
        removes = [(parameter, 0) for parameter in parameters]
        eq_separated = eq.subs(removes)
        eq_separated = sp.Eq(eq_separated.lhs, eq_separated.rhs + damping_function)
        return eq_separated

    def __repr__(self):

        s = (
            f"\n X_eq: \n {pretty(self.X_eq, use_unicode=False)} \n"
            + f"\n Y: \n {pretty(self.Y_eq, use_unicode=False)} \n"
            + f"\n N: \n {pretty(self.N_eq, use_unicode=False)} \n"
        )
        return s


class Simulator:
    def __init__(self, X_eq, Y_eq, N_eq):

        self.X_eq = X_eq
        self.Y_eq = Y_eq
        self.N_eq = N_eq

        self.acceleration_lambda = self.define_EOM(
            X_eq=self.X_eq, Y_eq=self.Y_eq, N_eq=self.N_eq
        )

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

    def define_EOM(self, X_eq: sp.Eq, Y_eq: sp.Eq, N_eq: sp.Eq):
        """Define equation of motion

        Args:
            X_eq (sp.Eq): [description]
            Y_eq (sp.Eq): [description]
            N_eq (sp.Eq): [description]
        """

        A, b = sp.linear_eq_to_matrix([X_eq, Y_eq, N_eq], [u1d, v1d, r1d])
        self.A = A
        self.b = b
        self.acceleartion_eq = A.inv() * b

        ## Lambdify:
        subs = {value: key for key, value in p.items()}
        subs[X_D] = sp.symbols("X_qs")
        subs[Y_D] = sp.symbols("Y_qs")
        subs[N_D] = sp.symbols("N_qs")

        acceleration_lambda = lambdify(self.acceleartion_eq.subs(subs))
        return acceleration_lambda

    def define_quasi_static_forces(
        self, X_qs_eq: sp.Eq, Y_qs_eq: sp.Eq, N_qs_eq: sp.Eq
    ):
        """Define the equations for the quasi static forces
        Ex:
        Y_qs(u,v,r,delta) = Yuu*u**2 + Yv*v + Yr*r + p.Ydelta*delta + ...

        Args:
            X_qs_eq (sp.Eq): [description]
            Y_qs_eq (sp.Eq): [description]
            N_qs_eq (sp.Eq): [description]
        """
        self.X_qs_eq = X_qs_eq
        self.Y_qs_eq = Y_qs_eq
        self.N_qs_eq = N_qs_eq

    def get_all_coefficients(self, sympy_symbols=True):
        return (
            self.get_coefficients_X(sympy_symbols=sympy_symbols)
            + self.get_coefficients_Y(sympy_symbols=sympy_symbols)
            + self.get_coefficients_N(sympy_symbols=sympy_symbols)
        )

    def get_coefficients_X(self, sympy_symbols=True):
        eq = self.X_eq.subs(X_D, self.X_qs_eq.rhs)
        return get_coefficients(eq=eq, sympy_symbols=sympy_symbols)

    def get_coefficients_Y(self, sympy_symbols=True):
        eq = self.Y_eq.subs(Y_D, self.Y_qs_eq.rhs)
        return get_coefficients(eq=eq, sympy_symbols=sympy_symbols)

    def get_coefficients_N(self, sympy_symbols=True):
        eq = self.N_eq.subs(N_D, self.N_qs_eq.rhs)
        return get_coefficients(eq=eq, sympy_symbols=sympy_symbols)

    def step(
        self,
        t: float,
        states: np.ndarray,
        parameters: dict,
        ship_parameters: dict,
        control: pd.DataFrame,
        U0=1,
    ) -> np.ndarray:
        """Calculate states derivatives for next time step


        Parameters
        ----------
        t : float
            current time
        states : np.ndarray
            current states as a vector
        parameters : dict
            hydrodynamic derivatives
        ship_parameters : dict
            ship parameters lpp, beam, etc.
        control : pd.DataFrame
            data frame with time series for control devices such as rudder angle (delta) and popeller thrust.
        U0 : float
            initial velocity constant [1] (only used for linearized models)

        Returns
        -------
        np.ndarray
            states derivatives for next time step
        """

        u, v, r, x0, y0, psi = states
        V = np.sqrt(u**2 + v**2)

        states_dict = {
            "u": u,
            "v": v,
            "r": r,
            "x0": x0,
            "y0": y0,
            "psi": psi,
        }

        inputs = dict(parameters)
        inputs.update(ship_parameters)
        inputs.update(states_dict)

        if isinstance(control, pd.DataFrame):
            index = np.argmin(np.array(np.abs(control.index - t)))
            control_ = dict(control.iloc[index])
        else:
            control_ = control

        rotation = R.from_euler("z", psi, degrees=False)
        w = 0
        velocities = rotation.apply([u, v, w])
        x01d = velocities[0]
        y01d = velocities[1]

        if "tws" in control_ and "twa" in control_:
            ## Calculate apparent wind:
            cog = np.arctan2(y01d, x01d)
            tws = control_.pop("tws")
            twa = control_.pop("twa")
            control_["aws"] = true_wind_speed_to_apparent(
                U=V, cog=cog, twa=twa, tws=tws
            )
            control_["awa"] = smallest_signed_angle(
                true_wind_angle_to_apparent(U=V, cog=cog, psi=psi, twa=twa, tws=tws)
            )

        inputs.update(control_)

        inputs["U"] = U0  # initial velocity constant [1]

        inputs["X_qs"] = run(function=self.X_qs_lambda, **inputs)
        inputs["Y_qs"] = run(function=self.Y_qs_lambda, **inputs)
        inputs["N_qs"] = run(function=self.N_qs_lambda, **inputs)
        u1d, v1d, r1d = run(function=self.acceleration_lambda, **inputs)

        # get rid of brackets:
        u1d = u1d[0]
        v1d = v1d[0]
        r1d = r1d[0]

        psi1d = r
        dstates = [
            u1d,
            v1d,
            r1d,
            x01d,
            y01d,
            psi1d,
        ]
        return dstates

    def step_primed_parameters(
        self, t, states, parameters, ship_parameters, control, U0
    ):
        """
        The simulation is carried out with states in SI units.
        The parameters are often expressed in prime system.
        This means that:
        1) the previous state needs to be converted to prime
        2) dstate is calculate in prime system
        3) dstate is converted back to SI.

        Args:
            t ([type]): [description]
            states ([type]): [description]
            parameters ([type]): [description]
            ship_parameters ([type]): [description]
            control ([type]): [description]
        """

        # 1)
        u, v, r, x0, y0, psi = states
        U = np.sqrt(u**2 + v**2)  # Instantanious velocity
        states_dict = {
            "u": u,
            "v": v,
            "r": r,
            "x0": x0,
            "y0": y0,
            "psi": psi,
        }
        states_dict_prime = self.prime_system.prime(states_dict, U=U)

        control_ = self.control(t=t, states=states, control=control)
        df_control_prime = self.prime_system.prime(control_, U=U)

        # 2)
        dstates_prime = self.step(
            t=t,
            states=list(states_dict_prime.values()),
            parameters=parameters,
            ship_parameters=self.ship_parameters_prime,
            control=df_control_prime,
            U0=1,
        )  # Note that U0 is 1 in prime system!

        # 3)
        (
            u1d_prime,
            v1d_prime,
            r1d_prime,
            x01d_prime,
            y01d_prime,
            psi1d_prime,
        ) = dstates_prime

        states_dict_prime = {
            "u1d": u1d_prime,
            "v1d": v1d_prime,
            "r1d": r1d_prime,
            "x01d": x01d_prime,
            "y01d": y01d_prime,
            "psi1d": psi1d_prime,
        }

        dstates_dict = self.prime_system.unprime(states_dict_prime, U=U)
        dstates = list(dstates_dict.values())
        return dstates

    def control(self, t: float, states: np.ndarray, control: dict) -> dict:
        """Controls, usually rudder angle and thrust
        (Override this method if thrust should also be simulated)

        Parameters
        ----------
        states : np.ndarray
            _description_
        control : dict
            'delta' : rudder angle [rad]
            'thrust': propeller thrust [N]

        Returns
        -------
        dict
            _description_
        """

        return control

    def simulate(
        self,
        df_,
        parameters,
        ship_parameters,
        control_keys=["delta", "thrust"],
        primed_parameters=False,
        prime_system=None,
        method="Radau",
        name="simulation",
        additional_events=[],
        include_accelerations=True,
        **kwargs,
    ):

        if not hasattr(self, "acceleration_lambda"):
            self.acceleration_lambda = self.define_EOM(
                X_eq=self.X_eq, Y_eq=self.Y_eq, N_eq=self.N_eq
            )

        t = df_.index
        t_span = [t.min(), t.max()]
        t_eval = np.linspace(t.min(), t.max(), len(t))

        df_control = df_[control_keys]

        self.primed_parameters = primed_parameters
        if primed_parameters:
            self.prime_system = prime_system
            assert isinstance(
                self.prime_system, vessel_manoeuvring_models.prime_system.PrimeSystem
            )
            self.ship_parameters_prime = self.prime_system.prime(ship_parameters)
            step = self.step_primed_parameters
        else:
            step = self.step

        # df_0 = df_.iloc[0:5].mean(axis=0)
        df_0 = df_.iloc[0]
        y0 = {
            "u": df_0["u"],
            "v": df_0["v"],
            "r": df_0["r"],
            "x0": df_0["x0"],
            "y0": df_0["y0"],
            "psi": df_0["psi"],
        }
        U0 = np.sqrt(df_0["u"] ** 2 + df_0["v"] ** 2)  # initial velocity constant [1]

        def stoped(t, states, parameters, ship_parameters, control, U0):
            u, v, r, x0, y0, psi = states
            return u

        stoped.terminal = True
        stoped.direction = -1

        def drifting(t, states, parameters, ship_parameters, control, U0):
            u, v, r, x0, y0, psi = states

            beta = np.deg2rad(70) - np.abs(-np.arctan2(v, u))

            return beta

        drifting.terminal = True
        drifting.direction = -1
        events = [stoped, drifting] + additional_events

        solution = solve_ivp(
            fun=step,
            t_span=t_span,
            y0=list(y0.values()),
            t_eval=t_eval,
            args=(parameters, ship_parameters, df_control, U0),
            method=method,
            events=events,
            **kwargs,
        )

        if not solution.success:
            # warnings.warn(solution.message)
            raise ValueError(solution.message)

        result = Result(
            simulator=self,
            solution=solution,
            df_model_test=df_,
            df_control=df_control,
            ship_parameters=ship_parameters,
            parameters=parameters,
            y0=y0,
            name=name,
            include_accelerations=include_accelerations,
        )
        return result

    @property
    def X_qs_lambda(self):

        if not hasattr(self, "_X_qs_lambda"):
            subs = {value: key for key, value in p.items()}
            self._X_qs_lambda = lambdify(self.X_qs_eq.rhs.subs(subs))

        return self._X_qs_lambda

    @property
    def Y_qs_lambda(self):

        if not hasattr(self, "_Y_qs_lambda"):
            subs = {value: key for key, value in p.items()}
            self._Y_qs_lambda = lambdify(self.Y_qs_eq.rhs.subs(subs))

        return self._Y_qs_lambda

    @property
    def N_qs_lambda(self):

        if not hasattr(self, "_N_qs_lambda"):
            subs = {value: key for key, value in p.items()}
            self._N_qs_lambda = lambdify(self.N_qs_eq.rhs.subs(subs))

        return self._N_qs_lambda


def get_coefficients(eq, sympy_symbols=True):
    coefficients = vessel_manoeuvring_models.models.diff_eq_to_matrix.get_coefficients(
        eq=eq,
        base_features=[
            u,
            v,
            r,
            delta,
            thrust,
            awa,
            aws,
            rho_A,
            A_XV,
            A_YV,
            L,
            # U,
            # tws,
            # twa,
        ],
    )
    if sympy_symbols:
        return coefficients
    else:
        subs = {value: key for key, value in p.items()}
        string_coefficients = [subs[coefficient] for coefficient in coefficients]
        return string_coefficients


class ModelSimulator(Simulator):
    """Ship and parameter specific simulator."""

    def __init__(
        self,
        simulator: Simulator,
        parameters: dict,
        ship_parameters: dict,
        control_keys: list,
        prime_system: PrimeSystem,
        name="simulation",
        primed_parameters=True,
        include_accelerations=True,
    ):
        """Generate a simulator that is specific to one ship with a specific set of parameters.
        This is done by making a copy of an existing simulator object and add freezed parameters.

        Parameters
        ----------
        simulator : Simulator
            Simulator object with predefined odes
        parameters : dict
            [description]
        ship_parameters : dict
            [description]
        control_keys : list
            [description]
        prime_system : PrimeSystem
            [description]
        name : str, optional
            [description], by default 'simulation'
        primed_parameters : bool, optional
            [description], by default True
        """

        self.__dict__.update(simulator.__dict__)
        self.parameters = self.extract_needed_parameters(parameters)
        self.ship_parameters = ship_parameters
        self.control_keys = control_keys
        self.primed_parameters = primed_parameters
        self.prime_system = prime_system
        self.ship_parameters_prime = self.prime_system.prime(ship_parameters)
        self.name = name
        self.include_accelerations = include_accelerations

    def copy(self):
        return deepcopy(self)

    def extract_needed_parameters(self, parameters: dict) -> dict:

        coefficients = self.get_all_coefficients(sympy_symbols=False)
        parameters = pd.Series(parameters).dropna()

        missing_coefficients = set(coefficients) - set(parameters.keys())
        assert (
            len(missing_coefficients) == 0
        ), f"Missing parameters:{missing_coefficients}"

        return parameters[coefficients]

    def simulate(
        self,
        df_,
        method="Radau",
        name="simulaton",
        additional_events=[],
        include_accelerations=True,
        **kwargs,
    ) -> Result:

        return super().simulate(
            df_=df_,
            parameters=self.parameters,
            ship_parameters=self.ship_parameters,
            control_keys=self.control_keys,
            primed_parameters=self.primed_parameters,
            prime_system=self.prime_system,
            method=method,
            name=name,
            additional_events=additional_events,
            include_accelerations=include_accelerations,
            **kwargs,
        )

    def turning_circle(
        self,
        u0: float,
        rev: float = None,
        angle: float = 35.0,
        t_max: float = 1000.0,
        dt: float = 0.01,
        method="Radau",
        name="simulation",
        **kwargs,
    ) -> Result:
        """Turning circle simulation

        Parameters
        ----------
        u0 : float
            initial speed [m/s]
        angle : float, optional
            Rudder angle [deg], by default 35.0 [deg]
        t_max : float, optional
            max simulation time, by default 1000.0
        dt : float, optional
            time step, by default 0.01, Note: The simulation time will not increase much with a smaller time step with Runge-Kutta!
        method : str, optional
            Method to solve ivp see solve_ivp, by default 'Radau'
        name : str, optional
            [description], by default 'simulation'

        Returns
        -------
        Result
            [description]
        """

        t_ = np.arange(0, t_max, dt)
        df_ = pd.DataFrame(index=t_)
        df_["x0"] = 0
        df_["y0"] = 0
        df_["psi"] = 0
        df_["u"] = u0
        df_["v"] = 0
        df_["r"] = 0
        assert np.abs(angle) > np.deg2rad(90), "angle should be in degrees!"

        df_["delta"] = np.deg2rad(angle)
        if not rev is None:
            df_["rev"] = rev

        def completed(t, states, parameters, ship_parameters, control, U0):
            u, v, r, x0, y0, psi = states
            remain = np.deg2rad(360) - np.abs(psi)
            return remain

        completed.terminal = True
        completed.direction = -1

        additional_events = [
            completed,
        ]

        return self.simulate(
            df_=df_,
            method=method,
            name=name,
            additional_events=additional_events,
            **kwargs,
        )

    def zigzag(
        self,
        u0: float,
        rev: float = None,
        angle: float = 10.0,
        heading_deviation: float = 10.0,
        t_max: float = 1000.0,
        dt: float = 0.01,
        rudder_rate=2.32,
        method="Radau",
        name="simulation",
        include_accelerations=True,
        **kwargs,
    ) -> Result:
        """ZigZag simulation

        Parameters
        ----------
        u0 : float
            initial speed [m/s]
        angle : float, optional
            Rudder angle [deg], by default 10.0 [deg]
        t_max : float, optional
            max simulation time, by default 1000.0
        dt : float, optional
            time step, by default 0.01, Note: The simulation time will not increase much with a smaller time step with Runge-Kutta!
        rudder_rate: float
            rudder rate [deg/s]
        method : str, optional
            Method to solve ivp see solve_ivp, by default 'Radau'
        name : str, optional
            [description], by default 'simulation'

        Returns
        -------
        Result
            [description]
        """

        t_ = np.arange(0, t_max, dt)
        df_ = pd.DataFrame(index=t_)
        df_["x0"] = 0
        df_["y0"] = 0
        df_["psi"] = 0
        df_["u"] = u0
        df_["v"] = 0
        df_["r"] = 0
        df_["delta"] = np.deg2rad(angle)

        if not rev is None:
            df_["rev"] = rev

        y0 = dict(df_.iloc[0])

        for control_key in self.control_keys:
            y0.pop(control_key)

        zig_zag_angle = np.abs(heading_deviation)
        direction = np.sign(angle)

        def course_deviated(t, states, parameters, ship_parameters, control, U0):
            u, v, r, x0, y0, psi = states
            target_psi = -direction * np.deg2rad(zig_zag_angle)
            remain = psi - target_psi
            return remain

        course_deviated.terminal = True

        additional_events = [
            course_deviated,
        ]

        results = []
        df_result = pd.DataFrame()

        ## 1)
        result = self.simulate(
            df_=df_,
            method=method,
            name=name,
            additional_events=additional_events,
            include_accelerations=False,
            **kwargs,
        )
        results.append(result)
        df_result = df_result.append(result.result)
        time = df_result.index[-1]

        ## 2)
        direction *= -1

        t_ = np.arange(time, time + t_max, dt)
        data = np.tile(df_result.iloc[-1], (len(t_), 1))
        df_ = pd.DataFrame(data=data, columns=df_result.columns, index=t_)

        t_local = t_ = np.arange(0, t_max, dt)
        delta_ = np.deg2rad(angle) + direction * np.deg2rad(rudder_rate) * t_local
        mask = np.abs(delta_) > np.deg2rad(np.abs(angle))
        delta_[mask] = direction * np.abs(np.deg2rad(angle))
        df_["delta"] = delta_

        #
        result = self.simulate(
            df_=df_,
            method=method,
            name=name,
            additional_events=additional_events,
            include_accelerations=False,
            **kwargs,
        )
        results.append(result)
        df_result = df_result.append(result.result.iloc[1:])
        time = df_result.index[-1]

        ## 3)
        direction *= -1

        t_ = np.arange(time, time + t_max, dt)
        data = np.tile(df_result.iloc[-1], (len(t_), 1))
        df_ = pd.DataFrame(data=data, columns=df_result.columns, index=t_)

        t_local = t_ = np.arange(0, t_max, dt)
        delta_ = (
            -direction * np.deg2rad(angle)
            + direction * np.deg2rad(rudder_rate) * t_local
        )
        mask = np.abs(delta_) > np.deg2rad(np.abs(angle))
        delta_[mask] = direction * np.abs(np.deg2rad(angle))
        df_["delta"] = delta_

        # df_["delta"] = direction * np.deg2rad(angle)

        #
        result = self.simulate(
            df_=df_,
            method=method,
            name=name,
            additional_events=additional_events,
            include_accelerations=False,
            **kwargs,
        )
        results.append(result)
        df_result = df_result.append(result.result.iloc[1:])
        time = df_result.index[-1]

        df_result_all = df_result[y0.keys()]

        solution_all = Bunch()
        solution_all.solution = results
        solution_all.y = df_result_all.values.T
        solution_all.t = np.array(df_result_all.index)

        result_all = Result(
            simulator=self,
            solution=solution_all,
            df_model_test=df_result,
            df_control=df_result[self.control_keys],
            ship_parameters=self.ship_parameters,
            parameters=self.parameters,
            include_accelerations=include_accelerations,
            name=name,
            y0=y0,
        )

        return result_all

    def forces(self, inputs: dict = {}) -> pd.DataFrame:
        """Get quasi static force from model

        Parameters
        ----------
        inputs : dict, optional
            innput states,  usually u,v,r, by default {}

        Returns
        -------
        pd.DataFrame
            Forces as a dataframe fx,fy,mz
        """

        inputs_prime = self.prime_system.prime(inputs, U=inputs["V"])

        outputs_prime = pd.DataFrame()
        dofs = ["fx", "fy", "mz"]
        for dof, func in zip(
            dofs, [self.X_qs_lambda, self.Y_qs_lambda, self.N_qs_lambda]
        ):
            outputs_prime[dof] = run(
                function=func,
                **inputs_prime,
                **self.parameters,
                **self.ship_parameters_prime,
            )

        outputs = self.prime_system.unprime(outputs_prime, U=inputs["V"])

        return outputs


class FullModelSimulator(ModelSimulator):
    def __init__(
        self,
        simulator: Simulator,
        parameters: dict,
        ship_parameters: dict,
        prime_system: PrimeSystem,
        model_pos: sm.regression.linear_model.RegressionResultsWrapper,
        model_neg: sm.regression.linear_model.RegressionResultsWrapper,
        propeller_coefficients: dict,
        control_keys: list = ["delta", "rev"],
        name="simulation",
        primed_parameters=True,
        include_accelerations=True,
    ):
        """Generate a simulator that is specific to one ship with a specific set of parameters.
        This is done by making a copy of an existing simulator object and add freezed parameters.

        Parameters
        ----------
        simulator : Simulator
            Simulator object with predefined odes
        parameters : dict
            [description]
        ship_parameters : dict
            [description]
        control_keys : list
            [description]
        prime_system : PrimeSystem
            [description]
        name : str, optional
            [description], by default 'simulation'
        primed_parameters : bool, optional
            [description], by default True
        """
        super().__init__(
            simulator=simulator,
            parameters=parameters,
            ship_parameters=ship_parameters,
            control_keys=control_keys,
            prime_system=prime_system,
            name=name,
            primed_parameters=primed_parameters,
            include_accelerations=include_accelerations,
        )
        self.model_pos = model_pos
        self.model_neg = model_neg
        self.propeller_coefficients = propeller_coefficients

    def control(self, t: float, states: np.ndarray, control: dict) -> dict:
        """Controls, usually rudder angle and thrust
        (Override this method if thrust should also be simulated)

        Parameters
        ----------
        states : np.ndarray
            _description_
        control : dict
            'delta' : rudder angle [rad]
            'thrust': propeller thrust [N]

        Returns
        -------
        dict
            _description_
        """

        u, v, r, x0, y0, psi = states
        index = np.argmin(np.array(np.abs(control.index - t)))
        control_ = dict(control.iloc[index])

        data = {
            "u": u,
            "v": v,
            "r": r,
            "x0": x0,
            "y0": y0,
            "psi": psi,
            "delta": control_["delta"],
            "rev": control_["rev"],
        }
        data = pd.Series(data, dtype=float)

        control["thrust"] = predictor(
            model_pos=self.model_pos,
            model_neg=self.model_neg,
            data=data,
            propeller_coefficients=self.propeller_coefficients,
            ship_data=self.ship_parameters,
        )

        return control


class ModelSimulatorWithPropeller(ModelSimulator):
    def __init__(
        self,
        simulator: Simulator,
        parameters: dict,
        ship_parameters: dict,
        prime_system: PrimeSystem,
        lambda_thrust,
        control_keys: list = ["delta", "rev"],
        name="simulation",
        primed_parameters=True,
        include_accelerations=True,
    ):
        """Generate a simulator that is specific to one ship with a specific set of parameters.
        This is done by making a copy of an existing simulator object and add freezed parameters.

        Parameters
        ----------
        simulator : Simulator
            Simulator object with predefined odes
        parameters : dict
            [description]
        ship_parameters : dict
            [description]
        control_keys : list
            [description]
        prime_system : PrimeSystem
            [description]
        lambda_thrust
            method that calculates the thrust, based on current state and parameters
        name : str, optional
            [description], by default 'simulation'
        primed_parameters : bool, optional
            [description], by default True
        """
        super().__init__(
            simulator=simulator,
            parameters=parameters,
            ship_parameters=ship_parameters,
            control_keys=control_keys,
            prime_system=prime_system,
            name=name,
            primed_parameters=primed_parameters,
            include_accelerations=include_accelerations,
        )
        self.lambda_thrust = lambda_thrust

    def control(self, t: float, states: np.ndarray, control: dict) -> dict:
        """Controls, usually rudder angle and thrust
        (Override this method if thrust should also be simulated)

        Parameters
        ----------
        states : np.ndarray
            _description_
        control : dict
            'delta' : rudder angle [rad]
            'thrust': propeller thrust [N]

        Returns
        -------
        dict
            _description_
        """

        u, v, r, x0, y0, psi = states
        index = np.argmin(np.array(np.abs(control.index - t)))
        control_ = dict(control.iloc[index])

        data = {
            "u": u,
            "v": v,
            "r": r,
            "x0": x0,
            "y0": y0,
            "psi": psi,
            "delta": control_["delta"],
            "rev": control_["rev"],
        }
        data["U"] = np.sqrt(data["u"] ** 2 + data["v"] ** 2)
        data = pd.Series(data, dtype=float)

        control["thrust"] = run(
            function=self.lambda_thrust,
            inputs=data,
            **self.ship_parameters,
            **self.parameters,
        )

        return control
