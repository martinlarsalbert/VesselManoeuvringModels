from vessel_manoeuvring_models.extended_kalman_filter import (
    extended_kalman_filter,
    rts_smoother,
)
import vessel_manoeuvring_models.extended_kalman_filter as ekf
from vessel_manoeuvring_models.symbols import *
from vessel_manoeuvring_models import prime_system
import sympy as sp
from vessel_manoeuvring_models.parameters import p
from copy import deepcopy
from vessel_manoeuvring_models.models.vmm import get_coefficients
from inspect import signature
import dill
from scipy.integrate import solve_ivp
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
from vessel_manoeuvring_models.substitute_dynamic_symbols import run
from scipy import interpolate

dill.settings["recurse"] = True

h = sp.symbols("h")  # time step


class SystemMatrixes:
    def __init__(self, vmm):
        """Define the system matrixes for a VMM
        Since this is time consuming and object of this class can be send to many ExtendedKalman objects.

        Parameters
        ----------
        vmm : _type_
            _description_

        """

        (
            self._lambda_f,
            self._lambda_jacobian,
            self.no_states,
            self.no_measurement_states,
        ) = define_system_matrixes_SI(
            vmm=vmm
        )  # (this one is slow)

    def copy(self):
        return deepcopy(self)

    def save(self, path: str):
        """Save model to pickle file

        Parameters
        ----------
        path : str
            Ex:'model.pkl'
        """

        with open(path, mode="wb") as file:
            dill.dump(self, file=file, recurse=True)

    def __getstate__(self):
        def should_pickle(k):
            return not k in []

        return {k: v for (k, v) in self.__dict__.items() if should_pickle(k)}

    @classmethod
    def load(cls, path: str):
        """Load model from pickle file

        Parameters
        ----------
        path : str
            Ex:'model.pkl'
        """

        with open(path, mode="rb") as file:
            obj = dill.load(file=file)

        return obj


class ExtendedKalman:
    """ExtendedKalman filter and smoother for a vessel manoeuvring model (vmm)"""

    def __init__(
        self,
        vmm,
        parameters: dict,
        ship_parameters: dict,
        system_matrixes: SystemMatrixes = None,
        demand_all_parameters=False,
    ):
        """ExtendedKalman filter and smoother for a Vessel Manoeuvring Model (VMM)

        Parameters
        ----------
        vmm : _type_
            Vessel Manoeuvring Model (VMM)
        parameters : dict
            hydrodynamic derivatives including added mass
        ship_parameters : dict
            ship parameters: lpp, mass,...
        system_matrixes : SystemMatrixes, optional
            Precalculated system matrixes can be provided to save time, these are otherwise calculated.
        demand_all_parameters : bool, optional
            Can we provide more parameters that the VMM needs?, by default False
        """

        self.X_eq = vmm.X_eq
        self.Y_eq = vmm.Y_eq
        self.N_eq = vmm.N_eq

        self.X_qs_eq = vmm.X_qs_eq
        self.Y_qs_eq = vmm.Y_qs_eq
        self.N_qs_eq = vmm.N_qs_eq

        self.parameters = self.extract_needed_parameters(
            parameters, demand_all_parameters
        )

        if system_matrixes is None:
            (
                self._lambda_f,
                self._lambda_jacobian,
                self.no_states,
                self.no_measurement_states,
            ) = define_system_matrixes_SI(
                vmm=vmm
            )  # (this one is slow)
        else:
            (
                self._lambda_f,
                self._lambda_jacobian,
                self.no_states,
                self.no_measurement_states,
            ) = (
                system_matrixes._lambda_f,
                system_matrixes._lambda_jacobian,
                system_matrixes.no_states,
                system_matrixes.no_measurement_states,
            )

        self.ship_parameters = ship_parameters
        self.needed_ship_parameters = self.extract_needed_ship_parameters(
            ship_parameters
        )

    def copy(self):
        return deepcopy(self)

    def save(self, path: str):
        """Save model to pickle file

        Parameters
        ----------
        path : str
            Ex:'model.pkl'
        """

        with open(path, mode="wb") as file:
            dill.dump(self, file=file, recurse=True)

    def __getstate__(self):
        def should_pickle(k):
            return not k in [
                "df_simulation",
                # "data",
                # "x0",
                # "P_prd",
                # "h",
                # "Qd",
                # "Rd",
                # "E",
                # "Cd",
                # "time_steps",
                # "time_steps_smooth",
            ]

        return {k: v for (k, v) in self.__dict__.items() if should_pickle(k)}

    @classmethod
    def load(cls, path: str):
        """Load model from pickle file

        Parameters
        ----------
        path : str
            Ex:'model.pkl'
        """

        with open(path, mode="rb") as file:
            obj = dill.load(file=file)

        return obj

    def extract_needed_parameters(
        self, parameters: dict, demand_all_parameters=False
    ) -> dict:
        coefficients = self.get_all_coefficients(sympy_symbols=False)
        parameters = pd.Series(parameters).dropna()

        missing_coefficients = set(coefficients) - set(parameters.keys())

        if demand_all_parameters:
            assert (
                len(missing_coefficients) == 0
            ), f"Missing parameters:{missing_coefficients}"
        else:
            replace = pd.Series(
                data=np.zeros(len(missing_coefficients)),
                index=list(missing_coefficients),
            )
            parameters = pd.concat(
                (parameters, replace)
            )  # Set missing coefficients to 0!

        return parameters[coefficients].copy()

    def extract_needed_ship_parameters(self, ship_parameters):
        s = signature(self._lambda_f)
        keys = list(set(ship_parameters) & set(s.parameters.keys()))
        new_ship_parameters = {
            key: value for key, value in ship_parameters.items() if key in keys
        }
        return new_ship_parameters

    def lambda_f(self, x, input: pd.Series) -> np.ndarray:
        # inputs = pd.Series(data=u, index=self.input_columns, dtype=float)

        psi = x[2]
        u = x[3]
        v = x[4]
        r = x[5]

        x_dot = self._lambda_f(
            **self.parameters,
            **self.needed_ship_parameters,
            **input,
            psi=psi,
            u=u,
            v=v,
            r=r,
        ).reshape(x.shape)

        return x_dot

    def lambda_jacobian(self, x: np.ndarray, input: pd.Series) -> np.ndarray:
        psi = x[2]
        u = x[3]
        v = x[4]
        r = x[5]

        jacobian = self._lambda_jacobian(
            **self.parameters,
            **self.needed_ship_parameters,
            **input,
            psi=psi,
            u=u,
            v=v,
            r=r,
            h=self.h,
        )
        return jacobian

    def simulate(
        self,
        x0_: np.ndarray = None,
        E: np.ndarray = None,
        ws: np.ndarray = None,
        data: pd.DataFrame = None,
        state_columns=["x0", "y0", "psi", "u", "v", "r"],
        input_columns=["delta"],
        solver="euler",
    ) -> pd.DataFrame:
        """Simulate with Euler forward integration where the state time derivatives are
        calculated using "lambda_f".

        This method is intended as a tool to study the system model that the
        Kalman filter is using. The simulation can be run with/without real data.

        with data:  "resimulation" : the simulation uses the same input as the real data
                    (specified by 'input_columns'). If 'x0' is not provided same initial
                    state is used.

        "without data": you need to create some "fake" data frame 'data' which contains
                        the necessary inputs. If 'x0' is not provided initial state must
                        be possible to take from this data frame also: by assigning the
                        initial state for all rows in the "fake" data frame.

        Parameters
        ----------
        x0 : np.ndarray, default None
            Initial state. If None, initial state is taken from first row of 'data'
        E : np.ndarray
            (no_states x no_hidden_states)
        ws : np.ndarray
            Process noise (no_time_stamps  x no_hidden_states)
        data : pd.DataFrame, default None
            Measured data can be provided
        input_columns : list
            what columns in 'data' are input signals?
        state_columns : list
            what colums in 'data' are the states?
        solver : default "euler"
                'euler"' euler forward method (very slow)
                or method passed to solve_ivp such as ‘Radau’ etc.


        Returns
        -------
        pd.DataFrame
            [description]
        """

        if data is None:
            assert hasattr(self, "data"), f"either specify 'data' or run 'filter' first"
        else:
            self.data = data
            self.input_columns = input_columns

        t = self.data.index

        # If None take value from object:
        if x0_ is None:
            if hasattr(self, "x0"):
                x0_ = self.x0
            else:
                x0_ = self.data.iloc[0][state_columns].values

        self.no_hidden_states = self.no_states - self.no_measurement_states

        if E is None:
            if not hasattr(self, "E"):
                self.E = np.vstack(
                    (
                        np.zeros((self.no_measurement_states, self.no_hidden_states)),
                        np.eye(self.no_hidden_states),
                    )
                )

            E = self.E

        if ws is None:
            ws = np.zeros((len(t), E.shape[1]))

        assert (
            len(x0_) == self.no_states
        ), f"length of 'x0' does not match the number of states ({self.no_states})"

        h = t[1] - t[0]
        Ed = h * E
        inputs = self.data[input_columns]

        if solver == "euler":
            simdata = self.euler_forward_integration(
                x0_=x0_, t=t, inputs=inputs, ws=ws, Ed=Ed
            )
        else:
            t_span = [t.min(), t.max()]
            res = solve_ivp(
                fun=self.step_in_time,
                t_span=t_span,
                y0=x0_,
                t_eval=t,
                method=solver,
                args=(ws, inputs, Ed),
            )
            simdata = res.y

        df = pd.DataFrame(
            simdata.T,
            columns=["x0", "y0", "psi", "u", "v", "r"],
            index=t[0 : simdata.shape[1]],
        )

        df.index.name = "time"
        df[self.input_columns] = inputs.iloc[0 : simdata.shape[1]].values

        self.df_simulation = df

        return df

    def euler_forward_integration(
        self,
        x0_,
        t,
        inputs,
        ws,
        Ed,
    ):
        simdata = np.zeros((len(x0_), len(t)))
        # x_ = x0_.reshape(len(x0_), 1)
        x_ = x0_

        h = t[1] - t[0]

        for i in range(len(t)):
            t_ = t[i]
            x_dot = self.step_in_time(t=t_, x_=x_, ws=ws, inputs=inputs, Ed=Ed)
            x_ = x_ + h * x_dot
            simdata[:, i] = x_.flatten()

        return simdata

    def step_in_time(self, t, x_, ws, inputs, Ed):
        x_ = x_.reshape(len(x_), 1)

        i = np.argmin(np.array(np.abs(inputs.index - t)))
        input = inputs.iloc[i]
        w_ = ws[i]

        w_ = w_.reshape(Ed.shape[1], 1)
        x_dot = self.lambda_f(x_, input) + Ed @ w_

        return x_dot.flatten()

    def filter(
        self,
        data: pd.DataFrame,
        P_prd: np.ndarray,
        Qd: float,
        Rd: float,
        E: np.ndarray,
        Cd: np.ndarray,
        state_columns=["x0", "y0", "psi", "u", "v", "r"],
        measurement_columns=["x0", "y0", "psi"],
        input_columns=["delta"],
        x0_: np.ndarray = None,
        do_checks=True,
    ) -> list:
        """kalman filter

        Parameters
        ----------
        x0_ : np.ndarray, default None
            initial state [x_1, x_2]
            The first row of the data is used as initial state if x0=None

        P_prd : np.ndarray
            initial covariance matrix (no_states x no_states)
        Qd : np.ndarray
            Covariance matrix of the process model (no_hidden_states x no_hidden_states)
        Rd : float
            Covariance matrix of the measurement (no_measurement_states x no_measurement_states)

        E: np.ndarray
            (no_states x no_hidden_states)

        Cd: np.ndarray
            (no_measurement_states  x no_states)
            Observation model selects the measurement states from all the states
            (Often referred to as H)

        state_columns : list
            what colums in 'data' are the states?

        measurement_columns: list
            name of columns in data that are measurements ex: ["x0", "y0", "psi"],

        input_columns: list
            name of columns in the data that are inputs ex: ["delta"]

        Returns
        -------
        list
            list with time steps as dicts.
        """

        self.data = data
        self.measurement_columns = measurement_columns
        self.input_columns = input_columns

        self.x0 = x0_
        if self.x0 is None:
            self.x0 = data.iloc[0][state_columns].values

        self.P_prd = np.array(P_prd)
        self.h = float(np.mean(np.diff(data.index)))
        self.Qd = np.array(Qd)
        self.Rd = np.array(Rd)
        self.E = E
        self.Cd = Cd

        time_steps = extended_kalman_filter(
            x0=self.x0,
            P_prd=self.P_prd,
            lambda_f=self.lambda_f,
            lambda_jacobian=self.lambda_jacobian,
            E=E,
            Qd=self.Qd,
            Rd=self.Rd,
            Cd=self.Cd,
            state_columns=state_columns,
            measurement_columns=measurement_columns,
            input_columns=input_columns,
            data=self.data,
            do_checks=do_checks,
        )

        self.time_steps = time_steps

        return time_steps

    def smoother(self, time_steps=None):
        if time_steps is None:
            assert hasattr(self, "x0"), "Please run 'filter' first"
            time_steps = self.time_steps

        time_steps_smooth = rts_smoother(
            time_steps=time_steps,
            lambda_jacobian=self.lambda_jacobian,
            Qd=self.Qd,
            lambda_f=self.lambda_f,
            E=self.E,
        )

        self.time_steps_smooth = time_steps_smooth
        return time_steps_smooth

    def get_all_coefficients(self, sympy_symbols=True):
        return list(
            set(
                self.get_coefficients_X(sympy_symbols=sympy_symbols)
                + self.get_coefficients_Y(sympy_symbols=sympy_symbols)
                + self.get_coefficients_N(sympy_symbols=sympy_symbols)
            )
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

    @property
    def x_hats(self):
        assert hasattr(self, "time_steps"), "Please run 'filter' first"
        return ekf.x_hat(self.time_steps)

    @property
    def time(self):
        assert hasattr(self, "time_steps"), "Please run 'filter' first"
        t0 = self.data.index[0]
        return t0 + ekf.time(self.time_steps)

    @property
    def inputs(self):
        assert hasattr(self, "time_steps"), "Please run 'filter' first"
        return ekf.inputs(self.time_steps)

    @property
    def variance(self):
        assert hasattr(self, "time_steps"), "Please run 'filter' first"
        return ekf.variance(self.time_steps)

    @property
    def kalman_gain(self):
        assert hasattr(self, "time_steps"), "Please run 'filter' first"
        return ekf.K(self.time_steps)

    @property
    def x_hats_smooth(self):
        assert hasattr(self, "time_steps_smooth"), "Please run 'smoother' first"
        return ekf.x_hat(self.time_steps_smooth)

    @property
    def time_smooth(self):
        assert hasattr(self, "time_steps_smooth"), "Please run 'smoother' first"
        t0 = self.data.index[0]
        return t0 + ekf.time(self.time_steps_smooth)

    @property
    def inputs_smooth(self):
        assert hasattr(self, "time_steps_smooth"), "Please run 'smoother' first"
        return ekf.inputs(self.time_steps_smooth)

    @property
    def variance_smooth(self):
        assert hasattr(self, "time_steps_smooth"), "Please run 'smoother' first"
        return ekf.variance(self.time_steps_smooth)

    def _df(self, x_hats, time):
        columns = list(self.model.states_str)
        df = pd.DataFrame(
            data=x_hats.T,
            index=time,
            columns=columns,
        )

        for key in ["u", "v", "r"]:
            key_acc = f"{key}1d"
            if not key_acc in df:
                df[key_acc] = np.gradient(df[key], df.index)
                columns.append(key_acc)

        ## Copy other data (interpolate to get same time)
        f = interpolate.interp1d(
            x=self.data.index,
            y=self.data.index,
            kind="nearest",
            bounds_error=False,
            fill_value="extrapolate",
        )
        data_copy_interpolated = self.data.loc[f(df.index)]
        data_copy_interpolated.index = df.index
        df = pd.concat((df, data_copy_interpolated.drop(columns=columns)), axis=1)

        ## Some signals should not take the nearest value, but rather the last known value:
        # target_signals = [
        #    "thrusterTarget",
        #    "courseTarget",
        #    "rudderTarget",
        # ]
        # for name in target_signals:
        #    # if name in self.data:
        #    last_known_value_interpolation(signal=self.data[name], data=df)
        #    df[name].fillna(0, inplace=True)

        ## Some signals should not be padded, but rather assign single values (the rest are NaN):
        one_value_signals = ["mission"]
        one_value_signals = list(set(one_value_signals) & set(self.data.columns))
        for name in one_value_signals:
            # if name in self.data:
            one_value_interpolation(signal=self.data[name], data=df)

        # copy_columns = list(
        #    set(self.data.select_dtypes(exclude="object").columns) - set(columns)
        # )
        # for column in copy_columns:
        #    f = interpolate.interp1d(
        #        x=self.data.index,
        #        y=self.data[column],
        #        kind="nearest",
        #        bounds_error=False,
        #        fill_value="extrapolate",
        #    )
        #    df[column] = f(df.index)

        # Updating these:
        df["V"] = df["U"] = np.sqrt(df["u"] ** 2 + df["v"] ** 2)
        df["beta"] = -np.arctan2(df["v"], df["u"])  # Drift angle

        return df

    @property
    def df_kalman(self):
        return self._df(x_hats=self.x_hats, time=self.time)

    @property
    def df_smooth(self):
        return self._df(x_hats=self.x_hats_smooth, time=self.time_smooth)

    @property
    def simulation_error(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """Provide time series of the error: simulation - data

        Parameters
        ----------
        data : pd.DataFrame, optional
            Real data, model test etc., by default None
            If None data is taken from object

        Returns
        -------
        pd.DataFrame
            time series with the error value: sim - data for each quantity in the states
        """

        assert hasattr(self, "df_simulation"), "Please run 'simulation' first"

        if data is None:
            assert hasattr(
                self, "data"
            ), "Please run 'simulation' first or provide 'data'"
            data = self.data

        error = self.df_simulation - data[self.df_simulation.columns].values
        return error


def last_known_value_interpolation(signal: pd.Series, data: pd.DataFrame):
    # one_value_interpolation(signal=signal, data=data)
    name = signal.name
    # data[name] = data[name].fillna(method="ffill").fillna(0)
    i = np.searchsorted(signal.index, data.index, side="right") - 1
    i = pd.Index(i)
    mask = i.isin(np.arange(0, len(signal)))
    good_i = i[mask]
    good_index = signal.index[good_i]
    ## Pad with initial NaNs if some of data.index are before signal.index
    number_of_NaN = (~mask).sum()
    nans = np.NaN * np.ones(number_of_NaN)

    interpolated = np.concatenate((nans, signal[good_index].values))
    data[signal.name] = interpolated


def one_value_interpolation(signal: pd.Series, data: pd.DataFrame, verify_unique=True):
    name = signal.name
    f = interpolate.interp1d(
        x=data.index,
        y=data.index,
        kind="nearest",
        bounds_error=False,
        fill_value="extrapolate",
    )
    index = f(signal.index)

    if verify_unique:
        assert pd.Index(
            index
        ).is_unique, "The interpolation has produced colliding indexes (data will be lost), set verify_unique=False to override this error"

    data[name] = np.NaN
    data.loc[index, name] = signal.values


class ExtendedKalmanModular(ExtendedKalman):
    def __init__(
        self,
        model: ModularVesselSimulator,
    ):
        """ExtendedKalman filter and smoother for a Vessel Manoeuvring Model (VMM)

        Parameters
        ----------
        model : ModularVesselSimulator
        """
        self.model = model
        self.X_eq = model.X_eq
        self.Y_eq = model.Y_eq
        self.N_eq = model.N_eq

        self.X_qs_eq = model.X_D_eq
        self.Y_qs_eq = model.Y_D_eq
        self.N_qs_eq = model.N_D_eq

        self.no_states = len(model.states)
        self.no_measurement_states = self.no_states - model.A.shape[0]
        self.ship_parameters = model.ship_parameters

    def lambda_f(self, x, input: pd.Series) -> np.ndarray:
        states_dict = {
            "x0": x[0],
            "y0": x[1],
            "psi": x[2],
            "u": x[3],
            "v": x[4],
            "r": x[5],
        }

        control = input
        calculation = self.model.calculate_forces(
            states_dict=states_dict, control=control
        )
        
        result =  self.model.lambda_f(
            **states_dict,
            **control,
            **self.model.parameters,
            **self.model.ship_parameters,
            **calculation,
            h=self.h,
            )
        
        #try:
        #    result =  self.model.lambda_f(
        #    **states_dict,
        #    **control,
        #    **self.model.parameters,
        #    **self.model.ship_parameters,
        #    **calculation,
        #    h=self.h,
        #    )
        #except:
        #    # slower...
        #    result =  run(
        #    self.model.lambda_f,
        #    inputs=states_dict,
        #    **control,
        #    **self.model.parameters,
        #    **self.model.ship_parameters,
        #    **calculation,
        #    h=self.h,
        #    )
        return result

    def lambda_jacobian(self, x, input: pd.Series) -> np.ndarray:
        states_dict = {
            "x0": x[0],
            "y0": x[1],
            "psi": x[2],
            "u": x[3],
            "v": x[4],
            "r": x[5],
        }

        control = input
        return self.model.calculate_jacobian(
            states_dict=states_dict, control=control, h=self.h
        )


def define_system_matrixes_SI(vmm):
    """Define the system matrixes in SI units
    This method generates two python methods and adds them as object attributes:
    "_lambda_f"
    dx/dt = f(u,v,r,...)
    _lambda_f(I_z, L, Ndelta, Nr, Nrdot, Nu, Nur, Nv, Nvdot, Xdelta, Xr, Xrr, Xu, Xudot, Xv, Xvr, Ydelta, Yr, Yrdot, Yu, Yur, Yv, Yvdot, delta, m, psi, r, rho, u, v, x_G)
    "_lambda_jacobian"
    df/dx
    _lambda_jacobian(I_z, L, Ndelta, Nr, Nrdot, Nu, Nur, Nv, Nvdot, Xdelta, Xr, Xrr, Xu, Xudot, Xv, Xvr, Ydelta, Yr, Yrdot, Yu, Yur, Yv, Yvdot, delta, h, m, psi, r, rho, u, v, x_G)
    """
    X_eq = vmm.X_eq
    Y_eq = vmm.Y_eq
    N_eq = vmm.N_eq
    A, b = sp.linear_eq_to_matrix([X_eq, Y_eq, N_eq], [u1d, v1d, r1d])
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
    subs = [
        (X_D, vmm.X_qs_eq.rhs),
        (Y_D, vmm.Y_qs_eq.rhs),
        (N_D, vmm.N_qs_eq.rhs),
    ]
    subs = subs + subs_prime
    A_SI = A.subs([(u, u_prime), (v, v_prime)]).subs(subs)
    b_SI = b.subs([(u, u_prime), (v, v_prime)]).subs(subs)
    x_dot = sympy.matrices.dense.matrix_multiply_elementwise(
        A_SI.inv() * b_SI,  # (Slow...)
        sp.Matrix(
            [
                (u**2 + v**2) / L,
                (u**2 + v**2) / L,
                (u**2 + v**2) / (L**2),
            ]
        ),
    )
    x_ = sp.Matrix(
        [u * sp.cos(psi) - v * sp.sin(psi), u * sp.sin(psi) + v * sp.cos(psi), r]
    )
    f_ = sp.Matrix.vstack(x_, x_dot)
    no_states = len(f_)
    no_measurement_states = no_states - A.shape[0]
    subs = {value: key for key, value in p.items()}
    subs[psi] = sp.symbols("psi")
    keys = list(set(subs.keys()) & set(f_.free_symbols))
    subs_ = {key: subs[key] for key in keys}
    expr = f_.subs(subs_)
    _lambda_f = sp.lambdify(list(expr.free_symbols), expr, modules="numpy")
    x, x1d = sp.symbols(r"\vec{x} \dot{\vec{x}}")  # State vector
    eq_x = sp.Eq(x, sp.UnevaluatedExpr(sp.Matrix([x_0, y_0, psi, u, v, r])))
    jac = sp.eye(6, 6) + f_.jacobian(eq_x.rhs.doit()) * h
    keys = list(set(subs.keys()) & set(jac.free_symbols))
    subs_ = {key: subs[key] for key in keys}
    expr = jac.subs(subs_)
    _lambda_jacobian = sp.lambdify(list(expr.free_symbols), expr, modules="numpy")

    return _lambda_f, _lambda_jacobian, no_states, no_measurement_states
