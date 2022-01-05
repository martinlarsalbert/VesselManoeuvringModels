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

        self.define_system_matrixes_SI()
        self.parameters = self.extract_needed_parameters(parameters)
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
            A_SI.inv() * b_SI,
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
        self._lambda_f = lambdify(f_.subs(subs))

        x, x1d = sp.symbols(r"\vec{x} \dot{\vec{x}}")  # State vector
        eq_x = sp.Eq(x, sp.UnevaluatedExpr(sp.Matrix([x_0, y_0, psi, u, v, r])))
        jac = sp.eye(6, 6) + f_.jacobian(eq_x.rhs.doit()) * h
        subs = {value: key for key, value in p.items()}
        subs[psi] = sp.symbols("psi")
        self._lambda_jacobian = lambdify(jac.subs(subs))

    def lambda_f(self, x, u) -> np.ndarray:
        delta = u

        psi = x[2]
        u = x[3]
        v = x[4]
        r = x[5]

        x_dot = run(
            self._lambda_f,
            **self.parameters,
            **self.ship_parameters,
            psi=psi,
            u=u,
            v=v,
            r=r,
            delta=delta,
        ).reshape(x.shape)
        return x_dot

    def lambda_jacobian(self, x, u) -> np.ndarray:

        delta = u

        psi = x[2]
        u = x[3]
        v = x[4]
        r = x[5]

        jacobian = run(
            self._lambda_jacobian,
            **self.parameters,
            **self.ship_parameters,
            psi=psi,
            u=u,
            v=v,
            r=r,
            delta=delta,
            h=self.h,
        )
        return jacobian

    def simulate(
        self,
        x0: np.ndarray,
        E: np.ndarray,
        ws: np.ndarray,
        t: np.ndarray,
        us: np.ndarray,
    ) -> pd.DataFrame:
        """Simulate Euler forward integration where the state time derivatives are
        calculated using "lambda_f".

        Parameters
        ----------
        x0 : np.ndarray
            Initial state
        E : np.ndarray
            (no_states x no_hidden_states)
        ws : np.ndarray
            Process noise (no_time_stamps  x no_hidden_states)
        t : np.ndarray
            Simulation time
        us : np.ndarray
            input signals (no_time_stamps  x no_inputs)

        Returns
        -------
        pd.DataFrame
            [description]
        """

        simdata = np.zeros((len(x0), len(t)))
        x_ = x0
        h = t[1] - t[0]
        Ed = h * E

        for i, (u_, w_) in enumerate(zip(us, ws)):

            w_ = w_.reshape(E.shape[1], 1)
            x_dot = self.lambda_f(x_, u_) + Ed @ w_
            x_ = x_ + h * x_dot

            simdata[:, i] = x_.flatten()

        df = pd.DataFrame(
            simdata.T, columns=["x0", "y0", "psi", "u", "v", "r"], index=t
        )
        df.index.name = "time"
        df["delta"] = us

        return df

    def filter(
        self,
        x0: np.ndarray,
        P_prd: np.ndarray,
        h: float,
        us: np.ndarray,
        ys: np.ndarray,
        Qd: float,
        Rd: float,
        E: np.ndarray,
        Cd: np.ndarray,
    ) -> list:
        """kalman filter

        Parameters
        ----------
        no_states : int
            number of states (same thing as for instance number of rows and cols in P_prd)

        no_measurement_states : int
            number of measurement states (same thing as for instance number of rows and cols in Rd)

        (no_hidden_states = no_states - no_measurement_states)

        x0 : np.ndarray
            initial state [x_1, x_2]

        P_prd : np.ndarray
            initial covariance matrix (no_states x no_states)
        h : float
            time step filter [s]
        us : np.ndarray
            1D array: inputs
        ys : np.ndarray
            1D array: measured yaw
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

        Returns
        -------
        list
            list with time steps as dicts.
        """

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
            us=us,
            ys=ys,
            E=E,
            Qd=Qd,
            Rd=Rd,
            Cd=Cd,
        )

        self.time_steps = time_steps

        return time_steps

    def smoother(self):

        assert hasattr(self, "x0"), "Please run 'filter' first"

        smooth_time_steps = rts_smoother(
            time_steps=self.time_steps,
            lambda_jacobian=self.lambda_jacobian,
            Qd=self.Qd,
            lambda_f=self.lambda_f,
            E=self.E,
        )

        self.smooth_time_steps = smooth_time_steps
        return smooth_time_steps

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
