from src.extended_kalman_filter import extended_kalman_filter, rts_smoother
from src.symbols import *
from src import prime_system
import sympy as sp
from src.parameters import p
from src.substitute_dynamic_symbols import lambdify, run
from copy import deepcopy
from src.models.vmm import get_coefficients

h = sp.symbols("h")  # time step


class ExtendedKalman:
    """ExtendedKalman filter and smoother for a vessel manoeuvring model (vmm)"""

    def __init__(self, vmm, parameters: dict, ship_parameters: dict):

        self.X_eq = vmm.X_eq
        self.Y_eq = vmm.Y_eq
        self.N_eq = vmm.N_eq

        self.X_qs_eq = vmm.X_qs_eq
        self.Y_qs_eq = vmm.Y_qs_eq
        self.N_qs_eq = vmm.N_qs_eq

        self.parameters = self.extract_needed_parameters(parameters)
        self.define_system_matrixes_SI()  # (this one is slow)
        self.ship_parameters = ship_parameters

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

    def define_system_matrixes_SI(self):
        """Define the system matrixes in SI units

        This method generates two python methods and adds them as object attributes:

        "_lambda_f"
        dx/dt = f(u,v,r,...)
        _lambda_f(I_z, L, Ndelta, Nr, Nrdot, Nu, Nur, Nv, Nvdot, Xdelta, Xr, Xrr, Xu, Xudot, Xv, Xvr, Ydelta, Yr, Yrdot, Yu, Yur, Yv, Yvdot, delta, m, psi, r, rho, u, v, x_G)

        "_lambda_jacobian"
        df/dx
        _lambda_jacobian(I_z, L, Ndelta, Nr, Nrdot, Nu, Nur, Nv, Nvdot, Xdelta, Xr, Xrr, Xu, Xudot, Xv, Xvr, Ydelta, Yr, Yrdot, Yu, Yur, Yv, Yvdot, delta, h, m, psi, r, rho, u, v, x_G)


        """

        X_eq = self.X_eq
        Y_eq = self.Y_eq
        N_eq = self.N_eq
        A, b = sp.linear_eq_to_matrix([X_eq, Y_eq, N_eq], [u1d, v1d, r1d])

        subs_prime = [
            (m, m / prime_system.df_prime.mass.denominator),
            (I_z, I_z / prime_system.df_prime.inertia_moment.denominator),
            (x_G, x_G / prime_system.df_prime.length.denominator),
            (u, u / sp.sqrt(u ** 2 + v ** 2)),
            (v, v / sp.sqrt(u ** 2 + v ** 2)),
            (r, r / (sp.sqrt(u ** 2 + v ** 2) / L)),
            (thrust, thrust / (1 / 2 * rho * (u ** 2 + v ** 2) * L ** 2)),
        ]

        subs = [
            (X_D, self.X_qs_eq.rhs),
            (Y_D, self.Y_qs_eq.rhs),
            (N_D, self.N_qs_eq.rhs),
        ]

        subs = subs + subs_prime

        A_SI = A.subs(subs)
        b_SI = b.subs(subs)

        self.x_dot = sympy.matrices.dense.matrix_multiply_elementwise(
            A_SI.inv() * b_SI,  # (Slow...)
            sp.Matrix(
                [
                    (u ** 2 + v ** 2) / L,
                    (u ** 2 + v ** 2) / L,
                    (u ** 2 + v ** 2) / (L ** 2),
                ]
            ),
        )

        x_ = sp.Matrix(
            [u * sp.cos(psi) - v * sp.sin(psi), u * sp.sin(psi) + v * sp.cos(psi), r]
        )

        f_ = sp.Matrix.vstack(x_, self.x_dot)
        self.no_states = len(f_)
        self.no_measurement_states = self.no_states - A.shape[0]

        subs = {value: key for key, value in p.items()}
        subs[psi] = sp.symbols("psi")

        keys = list(set(subs.keys()) & set(f_.free_symbols))
        subs_ = {key: subs[key] for key in keys}
        expr = f_.subs(subs_)
        self._lambda_f = sp.lambdify(list(expr.free_symbols), expr)

        x, x1d = sp.symbols(r"\vec{x} \dot{\vec{x}}")  # State vector
        eq_x = sp.Eq(x, sp.UnevaluatedExpr(sp.Matrix([x_0, y_0, psi, u, v, r])))
        jac = sp.eye(6, 6) + f_.jacobian(eq_x.rhs.doit()) * h

        keys = list(set(subs.keys()) & set(jac.free_symbols))
        subs_ = {key: subs[key] for key in keys}
        expr = jac.subs(subs_)
        self._lambda_jacobian = sp.lambdify(list(expr.free_symbols), expr)

    def lambda_f(self, x, input: pd.Series) -> np.ndarray:

        # inputs = pd.Series(data=u, index=self.input_columns, dtype=float)

        psi = x[2]
        u = x[3]
        v = x[4]
        r = x[5]

        x_dot = run(
            self._lambda_f,
            **self.parameters,
            **self.ship_parameters,
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

        jacobian = run(
            self._lambda_jacobian,
            **self.parameters,
            **self.ship_parameters,
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
        x0: np.ndarray = None,
        E: np.ndarray = None,
        ws: np.ndarray = None,
        data: pd.DataFrame = None,
        input_columns=["delta"],
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
        if x0 is None:
            if hasattr(self, "x0"):
                x0 = self.x0
            else:
                x0 = self.data.iloc[0][["x0", "y0", "psi", "u", "v", "r"]].values

        if E is None:
            assert hasattr(self, "E"), f"either specify 'E' or run 'filter' first"
            E = self.E

        if ws is None:
            ws = np.zeros((len(t), E.shape[1]))

        assert (
            len(x0) == self.no_states
        ), f"length of 'x0' does not match the number of states ({self.no_states})"

        simdata = np.zeros((len(x0), len(t)))
        x_ = x0.reshape(len(x0), 1)
        h = t[1] - t[0]
        Ed = h * E
        inputs = self.data[input_columns]

        for i in range(len(t)):

            input = inputs.iloc[i]
            w_ = ws[i]

            w_ = w_.reshape(E.shape[1], 1)
            x_dot = self.lambda_f(x_, input) + Ed @ w_
            x_ = x_ + h * x_dot

            simdata[:, i] = x_.flatten()

        df = pd.DataFrame(
            simdata.T, columns=["x0", "y0", "psi", "u", "v", "r"], index=t
        )
        df.index.name = "time"
        df[self.input_columns] = inputs.values

        self.df_simulation = df

        return df

    def filter(
        self,
        data: pd.DataFrame,
        P_prd: np.ndarray,
        Qd: float,
        Rd: float,
        E: np.ndarray,
        Cd: np.ndarray,
        measurement_columns=["x0", "y0", "psi"],
        input_columns=["delta"],
        x0: np.ndarray = None,
    ) -> list:
        """kalman filter

        Parameters
        ----------
        no_states : int
            number of states (same thing as for instance number of rows and cols in P_prd)

        no_measurement_states : int
            number of measurement states (same thing as for instance number of rows and cols in Rd)

        (no_hidden_states = no_states - no_measurement_states)

        x0 : np.ndarray, default None
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

        h = np.mean(np.diff(data.index))
        inputs = self.data[self.input_columns]
        self.ys = self.data[self.measurement_columns].values

        if x0 is None:
            x0 = self.data.iloc[0][["x0", "y0", "psi", "u", "v", "r"]].values

        self.x0 = x0
        self.P_prd = P_prd
        self.h = h
        self.Qd = Qd
        self.Rd = Rd
        self.E = E
        self.Cd = Cd

        time_steps = extended_kalman_filter(
            no_states=self.no_states,
            no_measurement_states=self.no_measurement_states,
            x0=x0,
            P_prd=P_prd,
            lambda_f=self.lambda_f,
            lambda_jacobian=self.lambda_jacobian,
            h=h,
            inputs=inputs,
            ys=self.ys,
            E=E,
            Qd=Qd,
            Rd=Rd,
            Cd=Cd,
        )

        self.time_steps = time_steps

        return time_steps

    def smoother(self):

        assert hasattr(self, "x0"), "Please run 'filter' first"

        time_steps_smooth = rts_smoother(
            time_steps=self.time_steps,
            lambda_jacobian=self.lambda_jacobian,
            Qd=self.Qd,
            lambda_f=self.lambda_f,
            E=self.E,
        )

        self.time_steps_smooth = time_steps_smooth
        return time_steps_smooth

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

    @staticmethod
    def _x_hat(time_steps):
        return np.array([time_step["x_hat"].flatten() for time_step in time_steps]).T

    @staticmethod
    def _time(time_steps):
        return np.array([time_step["time"] for time_step in time_steps]).T

    @staticmethod
    def _us(time_steps):
        return np.array([time_step["u"].flatten() for time_step in time_steps]).T

    @staticmethod
    def _variance(time_steps):
        return np.array([np.diagonal(time_step["P_hat"]) for time_step in time_steps]).T

    @property
    def x_hats(self):
        assert hasattr(self, "time_steps"), "Please run 'filter' first"
        return self._x_hat(self.time_steps)

    @property
    def time(self):
        assert hasattr(self, "time_steps"), "Please run 'filter' first"
        return self._time(self.time_steps)

    @property
    def us(self):
        assert hasattr(self, "time_steps"), "Please run 'filter' first"
        return self._us(self.time_steps)

    @property
    def variance(self):
        assert hasattr(self, "time_steps"), "Please run 'filter' first"
        return self._variance(self.time_steps)

    @property
    def x_hats_smooth(self):
        assert hasattr(self, "time_steps_smooth"), "Please run 'smoother' first"
        return self._x_hat(self.time_steps_smooth)

    @property
    def time_smooth(self):
        assert hasattr(self, "time_steps_smooth"), "Please run 'smoother' first"
        return self._time(self.time_steps_smooth)

    @property
    def us_smooth(self):
        assert hasattr(self, "time_steps_smooth"), "Please run 'smoother' first"
        return self._us(self.time_steps_smooth)

    @property
    def variance_smooth(self):
        assert hasattr(self, "time_steps_smooth"), "Please run 'smoother' first"
        return self._variance(self.time_steps_smooth)

    def _df(self, x_hats, time):

        df = pd.DataFrame(
            data=x_hats.T,
            index=time,
            columns=["x0", "y0", "psi", "u", "v", "r"],
        )

        for key in ["u", "v", "r"]:
            df[f"{key}1d"] = np.gradient(df[key], df.index)

        df[self.input_columns] = self.data[self.input_columns].values

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
