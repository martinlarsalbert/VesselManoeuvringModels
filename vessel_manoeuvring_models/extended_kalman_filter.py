import numpy as np
from numpy.linalg.linalg import inv, pinv
import pandas as pd
from typing import AnyStr, Callable
from copy import deepcopy
import numpy as np

from scipy.stats import multivariate_normal

from vessel_manoeuvring_models.angles import smallest_signed_angle
import logging

log = logging.getLogger(__name__)


def extended_kalman_filter(
    P_prd: np.ndarray,
    lambda_f: Callable,
    lambda_jacobian: Callable,
    data: pd.DataFrame,
    Qd: float,
    Rd: float,
    E: np.ndarray,
    Cd: np.ndarray,
    state_columns=["x0", "y0", "psi", "u", "v", "r"],
    measurement_columns=["x0", "y0", "psi"],
    input_columns=["delta"],
    x0: np.ndarray = None,
    angle_columns=["psi"],
    do_checks=True,
    **kwargs,
) -> list:
    """Example extended kalman filter

    Parameters
    ----------
    x0 : np.ndarray, default None
        initial state [x_1, x_2]
        The first row of the data is used as initial state if x0=None


    P_prd : np.ndarray
        initial covariance matrix (no_states x no_states)

    lambda_f: Callable
        python method that calculates the next time step

        Example:
        def lambda_f(x: np.ndarray, input: pd.Series) -> np.ndarray:

            b = 1
            w = 0

            u = input['delta]
            dx = np.array([[x[1], x[1] * np.abs(x[1]) + b * u + w]]).T

        the current state x and input are the only inputs to this method.
        Other parameters such as b and w in this example needs to be included as local
        variables in the method.

    lambda_jacobian: Callable

        python method that calculates the jacobian matrix

        Example:
        def lambda_jacobian(x: np.ndarray, input: pd.Series) -> np.ndarray:

            h=0.1

            jac = np.array(
                [
                    [1, h, 0],
                    [0, 2 * x[2] * h * np.abs(x[1]) + 1, h * x[1] * np.abs(x[1])],
                    [0, 0, 1],
                ]
            )
            return jac

        the current state x and input are the only inputs to this method.
        Other parameters such as time step h in this example needs to be included as local
        variables in the method.

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

    angle_columns: list
        name of columns that are angles ex: ["psi", "r"]
        The "smallest signed angle" are used for these states when the predictor error "eps" is calculated


    Returns
    -------
    list
        list with time steps as dicts.
    """

    no_states = len(state_columns)
    no_measurement_states = len(measurement_columns)

    no_hidden_states = no_states - no_measurement_states

    if x0 is None:
        x0 = data.iloc[0][state_columns].values

    ## Check dimensions:
    if do_checks:
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

    mask_angles = [key in angle_columns for key in measurement_columns]

    # Initial state:
    x_prd = np.array(x0).reshape(no_states, 1)
    P_prd = np.array(P_prd)

    time_steps = []

    inputs = data[input_columns]
    ys = data[measurement_columns]

    assert inputs.notnull().all().all(), "inputs contains NaNs"
    assert ys.notnull().all().all(), "measurements contains NaNs"

    h_median = np.median(np.diff(data.index))
    h = np.min(np.diff(data.index))
    if h_median / h > 4:
        log.warning(
            f"Using the smallest timestep in data will produce a lot of interations in Kalman filter."
        )

    ts = np.arange(
        data.index[0], data.index[-1], h
    )  # Time signal with equal time steps

    N = len(ts)
    Ed = h * E

    for t in ts:
        t_sample = find_closest(data=data, t=t)

        if abs(t - t_sample) < 0.7 * h:
            # Loading a measurement:
            input = inputs.loc[t_sample]  # input
            # y = np.array([ys[i]]).T  # measurement
            y = ys.loc[t_sample]  # measurement
            y = np.array([y.values]).T
            Cd_ = Cd
            # print("update")
        else:
            Cd_ = 0 * Cd  # no measurements
            # print("dead reckoning")

        # Compute kalman gain:
        # S = Cd @ P_prd @ Cd.T + Rd  # System uncertainty
        # K = P_prd @ Cd.T @ inv(S)
        try:
            K = P_prd @ Cd_.T @ pinv(Cd_ @ P_prd @ Cd_.T + Rd)
        except Exception as e:
            raise TypeError(f"Crashed at iteration time: {t} s") from e

        IKC = np.eye(no_states) - K @ Cd_

        ## State corrector:
        P_hat = IKC @ P_prd @ IKC.T + K @ Rd @ K.T
        eps = y - Cd_ @ x_prd
        eps[mask_angles] = smallest_signed_angle(
            eps[mask_angles]
        )  # Smalles signed angle
        x_hat = x_prd + K @ eps

        ## discrete-time extended KF-model
        f_hat = lambda_f(x=x_hat.flatten(), input=input).reshape((no_states, 1))

        ## Predictor (k+1)
        ## Ad = I + h * A and Ed = h * E
        ## where A = df/dx is linearized about x = x_hat
        Ad = lambda_jacobian(x=x_hat.flatten(), input=input)

        x_prd = x_hat + h * f_hat
        # x_prd = Ad @ x_hat + Bd * u  # (This gives a  linear discrete-time Kalman filter instead)

        P_prd = Ad @ P_hat @ Ad.T + Ed @ Qd @ Ed.T

        time_step = {
            "x_hat": x_hat,
            "P_hat": P_hat,
            "Ad": Ad,
            "time": t,
            "K": K,
            "eps": eps.flatten(),
            "input": input,
            "P_prd": P_prd,
            "x_prd": x_prd,
        }

        time_steps.append(time_step)

    return time_steps


def find_closest(data: pd.DataFrame, t: float) -> float:
    """Find closes time stamp

    Parameters
    ----------
    data : pd.DataFrame
        _description_
    t : float
        _description_

    Returns
    -------
    float
        _description_
    """
    i = np.argmin(np.abs(data.index - t))
    return data.index[i]


def rts_smoother(time_steps: list, lambda_jacobian: Callable, Qd, lambda_f, E):
    no_states = len(time_steps[0]["x_hat"])

    n = len(time_steps)

    s = deepcopy(time_steps)

    h = s[1]["time"] - s[0]["time"]
    Ed = h * E

    for k in range(n - 2, -1, -1):
        input = s[k]["input"]

        Ad = lambda_jacobian(x=s[k]["x_hat"].flatten(), input=input)
        # Pp = Ad @ s[k]["P_hat"] @ Ad.T + Qd  # predicted covariance
        Pp = Ad @ s[k]["P_hat"] @ Ad.T + Ed @ Qd @ Ed.T  # predicted covariance

        s[k]["K"] = s[k]["P_hat"] @ Ad.T @ pinv(Pp)

        # s[k]["x_hat"] += s[k]["K"] @ (s[k + 1]["x_hat"] - (Ad @ s[k]["x_hat"] + Bd * u))

        ## discrete-time extended KF-model
        x_hat = s[k]["x_hat"]
        f_hat = lambda_f(x=x_hat.flatten(), input=input).reshape((no_states, 1))
        x_prd = s[k]["x_hat"] + h * f_hat
        s[k]["x_prd"] = x_prd

        s[k]["x_hat"] += s[k]["K"] @ (s[k + 1]["x_hat"] - x_prd)
        s[k]["P_hat"] += s[k]["K"] @ (s[k + 1]["P_hat"] - Pp) @ s[k]["K"].T

    return s


def simulate(
    data: pd.DataFrame,
    lambda_f: Callable,
    E: np.ndarray = None,
    ws: np.ndarray = None,
    x0: np.ndarray = None,
    input_columns=["delta"],
    state_columns=["x0", "y0", "psi", "u", "v", "r"],
    hidden_state_columns=["u", "v", "r"],
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

    lambda_f: Callable
        python method that calculates the next time step

        Example:
        def lambda_f(x: np.ndarray, input: pd.Series) -> np.ndarray:

            b = 1
            w = 0

            u = input['delta]
            dx = np.array([[x[1], x[1] * np.abs(x[1]) + b * u + w]]).T

        the current state x and input are the only inputs to this method.
        Other parameters such as b and w in this example needs to be included as local
        variables in the method.

    E : np.ndarray, default None
        (no_states x no_hidden_states)
        None means that E is autogenerated

    ws : np.ndarray, default None
        Process noise (no_time_stamps  x no_hidden_states)
        None means no noise

    data : pd.DataFrame, default None
        Measured data can be provided

    input_columns : list
        what columns in 'data' are input signals?

    state_columns : list
        what colums in 'data' are the states?

    hidden_state_columns : list
        what columns are hidden state in 'data' (which has process noise)

    Returns
    -------
    pd.DataFrame
        [description]
    """

    t = data.index

    no_hidden_states = len(hidden_state_columns)
    no_states = len(state_columns)
    no_measurement_states = no_states - no_hidden_states
    assert (
        no_measurement_states >= 0
    ), f"Number of measurement states and hidden states does not add up to the total number of states"

    if E is None:
        E = np.vstack(
            (
                np.zeros((no_measurement_states, no_hidden_states)),
                np.eye(no_hidden_states),
            )
        )

    if ws is None:
        ws = np.zeros((len(t), no_hidden_states))

    if x0 is None:
        x0 = data.iloc[0][state_columns].values

    simdata = np.zeros((len(x0), len(t)))
    x_ = x0.reshape(len(x0), 1)
    h = t[1] - t[0]
    Ed = h * E
    inputs = data[input_columns]

    for i in range(len(t)):
        input = inputs.iloc[i]
        w_ = ws[i]

        w_ = w_.reshape(E.shape[1], 1)
        x_dot = lambda_f(x_.flatten(), input) + Ed @ w_
        x_ = x_ + h * x_dot

        simdata[:, i] = x_.flatten()

    df = pd.DataFrame(simdata.T, columns=state_columns, index=t)
    df.index.name = "time"
    df[input_columns] = inputs.values

    return df


def _loglikelihood(time_step: dict) -> float:
    mean = time_step["x_hat"].flatten()
    cov = time_step["P_hat"]
    x_prd = time_step["x_prd"]
    rv = multivariate_normal(mean=mean, cov=cov)
    return rv.logpdf(x_prd.flatten())


def loglikelihood(time_steps: list) -> float:
    """Calculate the log-likelihood of the time steps from the estimation

    Parameters
    ----------
    time_steps : list
        estimation time steps

    Returns
    -------
    float
        log-likelihood
    """

    loglikelihood = 0
    for time_step in time_steps:
        loglikelihood += _loglikelihood(time_step)

    return loglikelihood


def get_time_step_array(time_steps, key):
    return np.array([time_step[key].flatten() for time_step in time_steps]).T


def x_hat(time_steps):
    return get_time_step_array(time_steps, "x_hat")


def x_prd(time_steps):
    return get_time_step_array(time_steps, "x_prd")


def time(time_steps):
    return get_time_step_array(time_steps, "time").flatten()


def inputs(time_steps) -> pd.DataFrame:
    t = time(time_steps)
    return pd.DataFrame([time_step["input"] for time_step in time_steps], index=t)


def variance(time_steps):
    return np.array([np.diagonal(time_step["P_hat"]) for time_step in time_steps]).T


def K(time_steps):
    return np.array([np.diagonal(time_step["K"]) for time_step in time_steps]).T


def time_steps_to_df(
    time_steps: list,
    state_columns: list = ["x0", "y0", "psi", "u", "v", "r"],
    add_gradients=True,
) -> pd.DataFrame:
    x_hats = x_hat(time_steps)
    t = time(time_steps)
    inputs_ = inputs(time_steps)

    df = pd.DataFrame(
        data=x_hats.T,
        index=t,
        columns=state_columns,
    )

    if add_gradients:
        for key in np.array_split(state_columns, 2)[1]:
            df[f"{key}1d"] = np.gradient(df[key], df.index)

    df[inputs_.columns] = inputs_.values

    return df
