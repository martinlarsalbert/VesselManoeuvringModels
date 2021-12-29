import numpy as np
from numpy.linalg.linalg import inv, pinv
import pandas as pd
from typing import AnyStr, Callable
from copy import deepcopy


def extended_kalman_filter(
    no_states: int,
    no_measurement_states: int,
    x0: np.ndarray,
    P_prd: np.ndarray,
    lambda_f: Callable,
    lambda_jacobian: Callable,
    h: float,
    us: np.ndarray,
    ys: np.ndarray,
    Qd: float,
    Rd: float,
    E: np.ndarray,
    Cd: np.ndarray,
    Bd: np.ndarray,
) -> list:
    """Example extended kalman filter

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

    lambda_f: Callable
        python method that calculates the next time step

        Example:
        def lambda_f(x,u):

            b = 1
            w = 0

            x : states
            u : inputs
            dx = np.array([[x[1], x[1] * np.abs(x[1]) + b * u + w]]).T

        the current state x and input u are the only inputs to this method.
        Other parameters such as b and w in this example needs to be included as local
        variables in the method.

    lambda_jacobian: Callable

        python method that calculates the jacobian matrix

        Example:
        def lambda_jacobian(x, u):

            h=0.1

            jac = np.array(
                [
                    [1, h, 0],
                    [0, 2 * x[2] * h * np.abs(x[1]) + 1, h * x[1] * np.abs(x[1])],
                    [0, 0, 1],
                ]
            )
            return jac

        the current state x and input u are the only inputs to this method.
        Other parameters such as time step h in this example needs to be included as local
        variables in the method.

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
    no_hidden_states = no_states - no_measurement_states

    ## Check dimensions:
    assert (
        len(x0) == no_states
    ), f"This filter has {no_states} states ('no_states'), but initial state vector 'x0' should have length:{no_states}"

    assert P_prd.shape == (
        no_states,
        no_states,
    ), f"This filter has {no_states} states ('no_states'), the initial state covariance matrix ('P_prd') should have shape:{(no_states, no_states)}"

    assert Qd.shape == (
        no_hidden_states,
        no_hidden_states,
    ), f"This filter has {no_states} states ('no_states'), the Covariance matrix of the process model ('Qd') should have shape:{(no_hidden_states, no_hidden_states)}"

    assert Rd.shape == (
        no_measurement_states,
        no_measurement_states,
    ), f"This filter has {no_states} states ('no_states'), the Covariance matrix of the measurement ('Rd') should have shape:{(no_measurement_states, no_measurement_states)}"

    assert E.shape == (
        no_states,
        no_hidden_states,
    ), f"This filter has {no_states} states ('no_states'), ('E') should have shape:{(no_states,no_hidden_states)}"

    assert Cd.shape == (
        no_measurement_states,
        no_states,
    ), f"This filter has {no_states} states ('no_states'), ('Cd') should have shape:{(no_measurement_states,no_states)}"

    # Initial state:
    x_prd = np.array(x0).reshape(no_states, 1)
    P_prd = np.array(P_prd)

    time_steps = []

    N = len(us)
    Ed = h * E

    for i in range(N):
        t = i * h

        u = us[i]  # input
        y = np.array([ys[i]]).T  # measurement

        # Compute kalman gain:
        # S = Cd @ P_prd @ Cd.T + Rd  # System uncertainty
        # K = P_prd @ Cd.T @ inv(S)
        K = P_prd @ Cd.T @ pinv(Cd @ P_prd @ Cd.T + Rd)
        IKC = np.eye(no_states) - K @ Cd

        ## State corrector:
        P_hat = IKC @ P_prd @ IKC.T + K @ Rd @ K.T
        eps = y - Cd @ x_prd
        x_hat = x_prd + K @ eps

        ## discrete-time extended KF-model
        f_hat = lambda_f(x=x_hat.flatten(), u=u).reshape((no_states, 1))

        ## Predictor (k+1)
        ## Ad = I + h * A and Ed = h * E
        ## where A = df/dx is linearized about x = x_hat
        Ad = lambda_jacobian(x=x_hat.flatten(), u=u)

        # x_prd = x_hat + h * f_hat
        x_prd = Ad @ x_hat + Bd * u

        P_prd = Ad @ P_hat @ Ad.T + Ed @ Qd @ Ed.T

        time_step = {
            "x_hat": x_hat,
            "P_hat": P_hat,
            "Ad": Ad,
            "time": t,
            "K": K,
            "eps": eps.flatten(),
        }

        time_steps.append(time_step)

    return time_steps


def rts_smoother(time_steps: list, us: np.ndarray, lambda_jacobian: Callable, Qd, Bd):

    n = len(time_steps)

    s = deepcopy(time_steps)

    for k in range(n - 2, -1, -1):
        u = us[k]

        Ad = lambda_jacobian(x=s[k]["x_hat"], u=u)
        Pp = Ad @ s[k]["P_hat"] @ Ad.T + Qd  # predicted covariance

        s[k]["K"] = s[k]["P_hat"] @ Ad.T @ inv(Pp)

        s[k]["x_hat"] += s[k]["K"] @ (s[k + 1]["x_hat"] - (Ad @ s[k]["x_hat"] + Bd * u))

        s[k]["P_hat"] += s[k]["K"] @ (s[k + 1]["P_hat"] - Pp) @ s[k]["K"].T

    return s
