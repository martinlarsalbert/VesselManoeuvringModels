from scipy.integrate import solve_ivp
import sympy as sp
from vessel_manoeuvring_models.symbols import *
import dill
from sympy.printing import pretty

from vessel_manoeuvring_models.parameters import df_parameters
from vessel_manoeuvring_models.prime_system import PrimeSystem
from vessel_manoeuvring_models.substitute_dynamic_symbols import lambdify, run
from scipy.spatial.transform import Rotation as R
from vessel_manoeuvring_models.models.result import Result

p = df_parameters["symbol"]
subs_simpler = {value: key for key, value in p.items()}
subs_simpler[psi] = "psi"


class ModularVesselSimulator:
    def __init__(
        self,
        X_eq: sp.Eq,
        Y_eq: sp.Eq,
        N_eq: sp.Eq,
        parameters: dict,
        ship_parameters: dict,
        control_keys: list,
        states=[x_0, y_0, psi, u, v, r],
        do_create_jacobian=True,
    ):
        """Top level of a modular simulation, where the equations define
        the equation of motion as function of subcomponents:
        X_H : Hull
        X_R : Rudder
        X_P : Propeller
        etc.
        giving the top level equation:
        m*(\dot{u} - r**2*x_G - r*v) = X_{\dot{u}}*\dot{u} + X_H + X_R + X_P
        It is up to subcomponents to calculate X_H, X_R etc.

        Parameters
        ----------
        X_eq : sp.Eq
            Equation in X-direction
        Y_eq : sp.Eq
            Equation in Y-direction
        N_eq : sp.Eq
            Equation in N-direction

        """

        self.states = states

        self.X_eq = X_eq.copy()
        self.Y_eq = Y_eq.copy()
        self.N_eq = N_eq.copy()

        self.parameters = parameters
        self.ship_parameters = ship_parameters
        self.control_keys = control_keys
        self.prime_system = PrimeSystem(**ship_parameters)
        self.ship_parameters_prime = self.prime_system.prime(ship_parameters)

        self.X_D_eq = sp.Eq(
            X_D_, self.X_eq.subs([(m, 0), (I_z, 0), (u1d, 0), (v1d, 0), (r1d, 0)]).rhs
        )
        self.lambda_X_D = lambdify(self.X_D_eq.rhs, substitute_functions=True)

        self.Y_D_eq = sp.Eq(
            Y_D_, self.Y_eq.subs([(m, 0), (I_z, 0), (u1d, 0), (v1d, 0), (r1d, 0)]).rhs
        )
        self.lambda_Y_D = lambdify(self.Y_D_eq.rhs, substitute_functions=True)

        self.N_D_eq = sp.Eq(
            N_D_, self.N_eq.subs([(m, 0), (I_z, 0), (u1d, 0), (v1d, 0), (r1d, 0)]).rhs
        )
        self.lambda_N_D = lambdify(self.N_D_eq.rhs, substitute_functions=True)

        self.define_EOM()
        self.subsystems = {}

        if do_create_jacobian:
            self.create_predictor_and_jacobian()

    def __repr__(self):

        s = (
            f"\n X_eq: \n {pretty(self.X_eq, use_unicode=False)} \n"
            + f"\n Y: \n {pretty(self.Y_eq, use_unicode=False)} \n"
            + f"\n N: \n {pretty(self.N_eq, use_unicode=False)} \n"
        )
        return s

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

    def define_EOM(self):
        """Define equation of motion

        Args:
            X_eq (sp.Eq): [description]
            Y_eq (sp.Eq): [description]
            N_eq (sp.Eq): [description]
        """

        # X_eq_X_D = self.X_eq.subs(self.X_D_eq.rhs, X_D_)
        # Y_eq_Y_D = self.Y_eq.subs(self.Y_D_eq.rhs, Y_D_)
        # N_eq_N_D = self.N_eq.subs(self.N_D_eq.rhs, N_D_)
        X_eq_X_D = self.X_eq
        Y_eq_Y_D = self.Y_eq
        N_eq_N_D = self.N_eq

        A, b = sp.linear_eq_to_matrix([X_eq_X_D, Y_eq_Y_D, N_eq_N_D], [u1d, v1d, r1d])
        self.A = A
        self.b = b
        self.acceleartion_eq = sp.simplify(A.inv() * b)

        ## Rewrite in SI units:
        keys = [
            "Xudot",
            "Xvdot",
            "Xrdot",
            "Yudot",
            "Yvdot",
            "Yrdot",
            "Nudot",
            "Nvdot",
            "Nrdot",
        ]
        subs = {
            df_parameters.loc[key, "symbol"]: df_parameters.loc[key, "symbol"]
            * df_parameters.loc[key, "denominator"]
            for key in keys
        }
        self.acceleartion_eq_SI = sp.simplify(self.acceleartion_eq.subs(subs))

        ## Lambdify:
        ### First change to simpler symbols:
        subs = {value: key for key, value in p.items()}
        self.acceleration_lambda_SI = lambdify(
            self.acceleartion_eq_SI.subs(subs), substitute_functions=True
        )

        return self.acceleration_lambda_SI

    def create_predictor_and_jacobian(self):
        x_dot = self.acceleartion_eq_SI
        x_ = sp.Matrix(
            [u * sp.cos(psi) - v * sp.sin(psi), u * sp.sin(psi) + v * sp.cos(psi), r]
        )
        f_ = sp.Matrix.vstack(x_, x_dot)
        f_ = sympy.matrices.immutable.ImmutableDenseMatrix(f_)
        self.lambda_f = lambdify(f_.subs(subs_simpler), substitute_functions=True)

        self.jac = f_.jacobian(self.states)
        h = sp.symbols("h")  # Time step
        Phi = sp.eye(len(self.states), len(self.states)) + self.jac * h
        self.lambda_jacobian = lambdify(
            Phi.subs(subs_simpler), substitute_functions=True
        )
        return self.lambda_jacobian

    def calculate_forces(self, states_dict: dict, control: dict):

        calculation = {}
        for name, subsystem in self.subsystems.items():
            subsystem.calculate_forces(
                states_dict=states_dict, control=control, calculation=calculation
            )

        calculation["X_D"] = run(self.lambda_X_D, calculation)
        calculation["Y_D"] = run(self.lambda_Y_D, calculation)
        calculation["N_D"] = run(self.lambda_N_D, calculation)

        return calculation

    def calculate_acceleration(self, states_dict: dict, control: dict):

        calculation = self.calculate_forces(states_dict=states_dict, control=control)

        acceleration = run(
            self.acceleration_lambda_SI,
            inputs=states_dict,
            **control,
            **self.parameters,
            **self.ship_parameters,
            **calculation,
        )

        return acceleration

    def calculate_jacobian(self, states_dict: dict, control: dict, h: float):
        calculation = self.calculate_forces(states_dict=states_dict, control=control)
        for name, subsystem in self.subsystems.items():
            subsystem.calculate_partial_derivatives(
                states_dict=states_dict, control=control, calculation=calculation
            )

        jacobian_matrix = run(
            function=self.lambda_jacobian,
            inputs=states_dict,
            **control,
            **calculation,
            **self.ship_parameters,
            **self.parameters,
            h=h,
        )
        return jacobian_matrix

    def step(
        self,
        t: float,
        states: np.ndarray,
        control: pd.DataFrame,
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
        U : float
            ship velocity in [m/s] (used for prime system)

        Returns
        -------
        np.ndarray
            states derivatives for next time step
        """

        u, v, r, x0, y0, psi = states

        states_dict = {
            "u": u,
            "v": v,
            "r": r,
            "x0": x0,
            "y0": y0,
            "psi": psi,
        }

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

        acceleration = self.calculate_acceleration(
            states_dict=states_dict, control=control_
        )

        # get rid of brackets:
        u1d = acceleration[0][0]
        v1d = acceleration[1][0]
        r1d = acceleration[2][0]

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

    def simulate(
        self,
        df_,
        method="Radau",
        additional_events=[],
        **kwargs,
    ):

        t = df_.index
        t_span = [t.min(), t.max()]
        t_eval = df_.index

        self.df_control = df_[self.control_keys]

        df_0 = df_.iloc[0]
        self.y0 = {
            "u": df_0["u"],
            "v": df_0["v"],
            "r": df_0["r"],
            "x0": df_0["x0"],
            "y0": df_0["y0"],
            "psi": df_0["psi"],
        }

        def stoped(
            t,
            states,
            control,
        ):
            u, v, r, x0, y0, psi = states
            return u

        stoped.terminal = True
        stoped.direction = -1

        def drifting(
            t,
            states,
            control,
        ):
            u, v, r, x0, y0, psi = states

            beta = np.deg2rad(70) - np.abs(-np.arctan2(v, u))

            return beta

        drifting.terminal = True
        drifting.direction = -1
        events = [stoped, drifting] + additional_events

        self.solution = solve_ivp(
            fun=self.step,
            t_span=t_span,
            y0=list(self.y0.values()),
            t_eval=t_eval,
            args=(self.df_control,),
            method=method,
            events=events,
            **kwargs,
        )

        if not self.solution.success:
            # warnings.warn(solution.message)
            raise ValueError(self.solution.message)

        df_result = self.post_process_simulation()

        return df_result

    def post_process_simulation(self):

        columns = list(self.y0.keys())
        df_result = pd.DataFrame(
            data=self.solution.y.T, columns=columns, index=self.solution.t
        )

        # Forces:
        forces = pd.DataFrame(
            self.calculate_forces(
                states_dict=df_result[["u", "v", "r"]], control=self.df_control
            ),
            index=df_result.index,
        )

        # Acceleration
        acceleration = self.calculate_acceleration(
            states_dict=df_result[["u", "v", "r"]], control=self.df_control
        )
        acceleration = pd.DataFrame(
            acceleration.reshape(acceleration.shape[0], acceleration.shape[2]).T,
            index=df_result.index,
            columns=["u1d", "v1d", "r1d"],
        )

        df_result = pd.concat((df_result, acceleration, forces), axis=1)

        # Extra:
        df_result["beta"] = -np.arctan2(df_result["v"], df_result["u"])
        df_result["U"] = np.sqrt(df_result["u"] ** 2 + df_result["v"] ** 2)

        return df_result