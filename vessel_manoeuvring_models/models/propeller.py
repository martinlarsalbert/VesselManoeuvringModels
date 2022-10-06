import sympy as sp
from vessel_manoeuvring_models.symbols import *

eq_T = sp.Eq(thrust, rho * rev ** 2 * D ** 4 * K_T)
eq_K_T = sp.Eq(K_T, k_2 * J ** 2 + k_1 * J + k_0)
eq_J = sp.Eq(J, u * (1 - w_p) / (rev * D))

eqs = [
    eq_T,
    eq_J,
    eq_K_T,
]

# Thrust
solution = sp.solve(eqs, thrust, K_T, J, dict=True)[0][thrust]
eq_thrust_simple = sp.Eq(thrust, solution)
lambda_thrust_simple = sp.lambdify(
    list(eq_thrust_simple.rhs.free_symbols), eq_thrust_simple.rhs
)

# w_p
solution = sp.solve(eq_thrust_simple, w_p, dict=True)[1][w_p]
eq_w_p = sp.Eq(w_p, solution)
lambda_w_p = sp.lambdify(list(eq_w_p.rhs.free_symbols), eq_w_p.rhs)

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
