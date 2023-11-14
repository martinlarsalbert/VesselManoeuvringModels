import sympy as sp
from vessel_manoeuvring_models.symbols import *

eq_T = sp.Eq(thrust_propeller, rho * rev**2 * D**4 * K_T)
eq_Q = sp.Eq(torque_propeller, rho * rev**2 * D**5 * K_Q / eta_r)
eq_K_T = sp.Eq(K_T, k_2 * J**2 + k_1 * J + k_0)
eq_K_Q = sp.Eq(K_Q, k_q1 * J + k_q0)
eq_J = sp.Eq(J, u * (1 - w_p) / (rev * D))
eq_η0 = sp.Eq(
    η0,
    sp.simplify(
        (eq_T.rhs * u / (eq_Q.rhs * 2 * sp.pi * rev))
        .subs(rev, sp.solve(eq_J, rev)[0])
        .subs(w_p, 0)
        .subs(eta_r, 1)
    ),
)
eq_P_d = sp.Eq(P_d, n_prop * 2 * sp.pi * rev * eq_Q.rhs)


eqs = [
    eq_T,
    eq_J,
    eq_K_T,
]

# Thrust
solution = sp.solve(eqs, thrust_propeller, K_T, J, dict=True)[0][thrust_propeller]
eq_thrust_simple = sp.Eq(thrust_propeller, solution)
lambda_thrust_simple = sp.lambdify(
    list(eq_thrust_simple.rhs.free_symbols), eq_thrust_simple.rhs
)

# w_p
solution = sp.solve(eq_thrust_simple, w_p, dict=True)[1][w_p]
eq_w_p = sp.Eq(w_p, solution)
lambda_w_p = sp.lambdify(list(eq_w_p.rhs.free_symbols), eq_w_p.rhs)

C0_w_p0, C1_w_p0, F_n = sp.symbols("C0_w_p0, C1_w_p0,F_n")
eq_w_p0 = sp.Eq(w_p0, C0_w_p0 + C1_w_p0 * F_n)
eq_F_n = sp.Eq(F_n, U / sp.sqrt(L * g))
eq_w_p0 = eq_w_p0.subs(F_n, eq_F_n.rhs)

from vessel_manoeuvring_models.models.subsystem import EquationSubSystem
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator


class PropellerSystem(EquationSubSystem):
    def __init__(
        self, ship: ModularVesselSimulator, create_jacobians=True, suffix="port"
    ):
        from vessel_manoeuvring_models.parameters import df_parameters

        suffix_str = f"_{suffix}" if len(suffix) > 0 else ""

        p = df_parameters["symbol"]

        C0_w_p0, C1_w_p0, F_n = sp.symbols("C0_w_p0, C1_w_p0,F_n")
        eq_w_p0 = sp.Eq(w_p0, C0_w_p0 + C1_w_p0 * F_n)
        eq_F_n = sp.Eq(F_n, U / sp.sqrt(L * g))
        eq_w_p0 = eq_w_p0.subs(F_n, eq_F_n.rhs)
        # Assuming that is a good model:
        eq_w_p = eq_w_p0.subs(w_p0, w_p)

        eqs = [eq_T, eq_K_T, eq_J, eq_w_p]
        solution = sp.solve(eqs, thrust_propeller, K_T, J, w_p, dict=True)
        eq = sp.Eq(thrust, solution[0][thrust_propeller])

        eq_X_P = sp.Eq(X_P, p.Xthrust * thrust)
        eq_Y_P = sp.Eq(Y_P, 0)
        eq_N_P = sp.Eq(N_P, -y_p * X_P).subs(y_p, f"y_p{suffix_str}")

        equations = [eq, eq_X_P, eq_Y_P, eq_N_P]

        if len(suffix) > 0:
            # Adding a suffix to distinguish between port and starboard rudder
            subs = {eq.lhs: f"{eq.lhs}{suffix_str}" for eq in equations}
            equations = [eq.subs(subs) for eq in equations]

        super().__init__(
            ship=ship, equations=equations, create_jacobians=create_jacobians
        )


X_P_port, Y_P_port, N_P_port, X_P_stbd, Y_P_stbd, N_P_stbd = sp.symbols(
    "X_P_port, Y_P_port, N_P_port, X_P_stbd, Y_P_stbd, N_P_stbd"
)


class PropellersSystem(EquationSubSystem):
    def __init__(
        self, ship: ModularVesselSimulator, create_jacobians=True, suffix="port"
    ):
        eq_X_P = sp.Eq(X_P, X_P_port + X_P_stbd)
        eq_Y_P = sp.Eq(Y_P, Y_P_port + Y_P_stbd)
        eq_N_P = sp.Eq(N_P, N_P_port + N_P_stbd)

        equations = [eq_X_P, eq_Y_P, eq_N_P]

        super().__init__(
            ship=ship, equations=equations, create_jacobians=create_jacobians
        )


class PropellersSimpleSystem(EquationSubSystem):
    def __init__(self, ship: ModularVesselSimulator, create_jacobians=True):
        from vessel_manoeuvring_models.parameters import df_parameters

        p = df_parameters["symbol"]
        eq_X_P_port = sp.Eq(X_P_port, p.Xthrustport * thrust_port)
        eq_X_P_stbd = sp.Eq(X_P_stbd, p.Xthruststbd * thrust_stbd)
        eq_X_P = sp.Eq(X_P, X_P_port + X_P_stbd)
        eq_Y_P = sp.Eq(Y_P, 0)
        eq_N_P_port = sp.Eq(N_P_port, -y_p_port * X_P_port)
        eq_N_P_stbd = sp.Eq(N_P_stbd, -y_p_stbd * X_P_stbd)
        eq_N_P = sp.Eq(N_P, N_P_port + N_P_stbd)
        # eq_thrust = sp.Eq(thrust, thrust_port + thrust_stbd)

        equations = [
            eq_X_P_port,
            eq_X_P_stbd,
            eq_X_P,
            eq_Y_P,
            eq_N_P_port,
            eq_N_P_stbd,
            eq_N_P,
            # eq_thrust,
        ]
        super().__init__(
            ship=ship, equations=equations, create_jacobians=create_jacobians
        )


class PropellerSimpleSystem(EquationSubSystem):
    def __init__(self, ship: ModularVesselSimulator, create_jacobians=True):
        from vessel_manoeuvring_models.parameters import df_parameters

        p = df_parameters["symbol"]
        eq_X_P_port = sp.Eq(X_P_port, p.Xthrustport * thrust_port)
        eq_X_P_stbd = sp.Eq(X_P_stbd, p.Xthruststbd * thrust_stbd)
        eq_X_P = sp.Eq(X_P, X_P_port + X_P_stbd)
        eq_Y_P = sp.Eq(Y_P, 0)
        eq_N_P_port = sp.Eq(N_P_port, -y_p_port * X_P_port)
        eq_N_P_stbd = sp.Eq(N_P_stbd, -y_p_stbd * X_P_stbd)
        eq_N_P = sp.Eq(N_P, N_P_port + N_P_stbd)
        # eq_thrust = sp.Eq(thrust, thrust_port + thrust_stbd)

        eq_X_P = sp.Eq(X_P, p.Xthrust * thrust)
        eq_Y_P = sp.Eq(Y_P, 0)
        eq_N_P = sp.Eq(N_P, 0)

        equations = [
            eq_X_P,
            eq_Y_P,
            eq_N_P,
        ]
        super().__init__(
            ship=ship, equations=equations, create_jacobians=create_jacobians
        )


import numpy as np
import pandas as pd
from vessel_manoeuvring_models.substitute_dynamic_symbols import run
from vessel_manoeuvring_models.prime_system import PrimeSystem
import statsmodels.api as sm


def preprocess(
    data: pd.DataFrame, ship_data: dict, propeller_coefficients: dict
) -> pd.DataFrame:
    data["beta"] = -np.arctan2(data["v"], data["u"])
    ps = PrimeSystem(**ship_data)
    xp_prime = ps._prime(
        ship_data["x_p"],
        unit="length",
    )
    data["U"] = np.sqrt(data["u"] ** 2 + data["v"] ** 2)
    r_prime = ps._prime(data["r"], unit="angular_velocity", U=data["U"])
    data["beta_p"] = data["beta"] - xp_prime * r_prime

    return data


def features(df, ship_data: dict, add_constant=False):
    if isinstance(df, pd.DataFrame):
        X = pd.DataFrame(index=df.index.copy())
    else:
        X = pd.DataFrame(index=[0])

    twin = ship_data["TWIN"] == 1
    if twin:
        X["delta"] = 0
    else:
        X["delta"] = df["delta"]

    X["delta**2"] = df["delta"] ** 2
    v_p = df["v"] + df["r"] * ship_data["x_p"]

    # if twin:
    #    X["beta_p"] = 0
    # else:
    #    X["beta_p"] = df["beta_p"]

    X["beta_p**2"] = df["beta_p"] ** 2
    X["u"] = df["u"]

    # X["v_p**2"] = v_p ** 2

    if add_constant:
        X = sm.tools.add_constant(X, has_constant="add")

    return X


def fit(
    data: pd.DataFrame,
    ship_data: dict,
    propeller_coefficients: dict,
    add_constant=False,
):
    data = preprocess(
        data, ship_data=ship_data, propeller_coefficients=propeller_coefficients
    )

    if ship_data["TWIN"] == 1:
        data = data.copy()
        data["thrust"] /= 2
    data["w_p"] = run(lambda_w_p, inputs=data, **ship_data, **propeller_coefficients)

    mask = data["beta_p"] > 0
    df_pos = data.loc[mask].copy()

    X = features(df_pos, ship_data=ship_data, add_constant=add_constant)
    y = df_pos["w_p"] - ship_data["w_p0"]
    linear_regression_pos = sm.OLS(y, X, hasconst=add_constant)
    model_pos = linear_regression_pos.fit()

    mask = data["beta_p"] <= 0
    df_neg = data.loc[mask].copy()

    X = features(df_neg, ship_data=ship_data, add_constant=add_constant)
    y = df_neg["w_p"] - ship_data["w_p0"]
    linear_regression_neg = sm.OLS(y, X, hasconst=add_constant)
    model_neg = linear_regression_neg.fit()

    return model_pos, model_neg


def predict(
    model_pos: sm.regression.linear_model.RegressionResultsWrapper,
    model_neg: sm.regression.linear_model.RegressionResultsWrapper,
    data: pd.DataFrame,
    propeller_coefficients: dict,
    ship_data: dict,
) -> pd.DataFrame:
    """Predict thrust for many time steps

    Parameters
    ----------
    model_pos : sm.regression.linear_model.RegressionResultsWrapper
        _description_
    model_neg : sm.regression.linear_model.RegressionResultsWrapper
        _description_
    data : pd.DataFrame
        _description_
    propeller_coefficients : dict
        _description_
    ship_data : dict
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """

    data = preprocess(
        data, ship_data=ship_data, propeller_coefficients=propeller_coefficients
    )

    add_constant = "const" in model_pos.params
    X = features(data, ship_data=ship_data, add_constant=add_constant)
    df_result = data.copy()

    df_result["w_p"] = (
        np.where(data["beta_p"] > 0, model_pos.predict(X), model_neg.predict(X))
        + ship_data["w_p0"]
    )
    df_result["thrust"] = run(
        lambda_thrust_simple, inputs=df_result, **propeller_coefficients, **ship_data
    )

    if ship_data["TWIN"] == 1:
        df_result["thrust"] *= 2

    return df_result


def predictor(
    model_pos: sm.regression.linear_model.RegressionResultsWrapper,
    model_neg: sm.regression.linear_model.RegressionResultsWrapper,
    data: dict,
    propeller_coefficients: dict,
    ship_data: dict,
) -> dict:
    """Predict thrust for ONE time step.

    Parameters
    ----------
    model_pos : sm.regression.linear_model.RegressionResultsWrapper
        _description_
    model_neg : sm.regression.linear_model.RegressionResultsWrapper
        _description_
    data : dict
        _description_
    propeller_coefficients : dict
        _description_
    ship_data : dict
        _description_

    Returns
    -------
    float
        thrust [N]
    """

    data = preprocess(
        data, ship_data=ship_data, propeller_coefficients=propeller_coefficients
    )

    add_constant = "const" in model_pos.params
    # print(data)
    X = features(data, ship_data=ship_data, add_constant=add_constant)
    # print(X)
    df_result = data.copy()

    beta_p = data["beta_p"]
    if beta_p > 0:
        model = model_pos
    else:
        model = model_neg

    df_result["w_p"] = model.predict(X)[0] + ship_data["w_p0"]

    thrust = run(
        lambda_thrust_simple, inputs=df_result, **propeller_coefficients, **ship_data
    )

    if ship_data["TWIN"] == 1:
        thrust *= 2

    return thrust
