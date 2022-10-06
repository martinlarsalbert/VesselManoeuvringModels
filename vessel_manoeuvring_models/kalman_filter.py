import numpy as np
import numpy as np
from numpy.linalg.linalg import inv, pinv
import pandas as pd
from typing import AnyStr, Callable


def ssa(angle):
    """
    maps an angle in rad to the interval [-pi pi]
    """
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi


def filter_yaw(
    x0: np.ndarray,
    P_prd: np.ndarray,
    h_m: float,
    h: float,
    us: np.ndarray,
    ys: np.ndarray,
    Ad: np.ndarray,
    Bd: np.ndarray,
    Cd: np.ndarray,
    Ed: np.ndarray,
    Qd: float,
    Rd: float,
) -> list:
    """kalman filter for yaw

    Parameters
    ----------
    x0 : np.ndarray
        initial state [yaw, yaw rate]
    P_prd : np.ndarray
        nxn array: initial covariance matrix
    h_m : float
        time step measurement [s]
    h : float
        time step filter [s]
    us : np.ndarray
        1D array: inputs
    ys : np.ndarray
        1D array: measured yaw
    Ad : np.ndarray
        nxn array: discrete time transition matrix
    Bd : np.ndarray
        nx1 array: discrete time input transition matrix
    Cd : np.ndarray
        1xn array: measurement transition matrix
    Ed : np.ndarray
        nx1 array
    Qd : float
        process noise
    Rd : float
        measurement noise

    Returns
    -------
    list
        list with data from each time step as dictionary
    """
    x_prd = x0
    t = 0
    n = len(Ad)  # Number of states

    time_steps = []
    for i in range(len(us)):

        u = us[i]  # input
        y = ys[i].T  # measurement

        for j in range(int(h_m / h)):
            t += h
            # Compute kalman gain:
            S = Cd @ P_prd @ Cd.T + Rd  # System uncertainty
            K = P_prd @ Cd.T @ inv(S)
            IKC = np.eye(n) - K @ Cd

            # State corrector:
            x_hat = x_prd + K * ssa(y - Cd @ x_prd)  # smallest signed angle

            # corrector
            P_hat = IKC * P_prd @ IKC.T + K * Rd @ K.T

            # Predictor (k+1)
            x_prd = Ad @ x_hat + Bd * u
            P_prd = Ad @ P_hat @ Ad.T + Ed @ Qd @ Ed.T

            time_step = {
                "x_hat": x_hat.flatten(),
                "P_hat": P_hat,
                "time": t,
                "K": K.flatten(),
            }
            time_steps.append(time_step)

    return time_steps


def rts_smoother(x_hats, P_hats, Ad, Bd, Qd, us):

    n, dim_x = x_hats.shape

    Bd = Bd.flatten()

    # smoother gain
    K = np.zeros((n, dim_x, dim_x))
    x, P, Pp = x_hats.copy(), P_hats.copy(), P_hats.copy()

    for k in range(n - 2, -1, -1):
        Pp[k] = Ad @ P[k] @ Ad.T + Qd  # predicted covariance

        K[k] = P[k] @ Ad.T @ inv(Pp[k])
        # x[k] += K[k] @ (x[k + 1] - (Ad @ x[k]))
        x[k] += K[k] @ (
            x[k + 1] - (Ad @ x[k] + Bd * us[k])
        )  # (unsure if us is introduced correctly)

        P[k] += K[k] @ (P[k + 1] - Pp[k]) @ K[k].T
    return (x, P, K, Pp)


def filter_yaw_example(
    x0: np.ndarray,
    P_prd: np.ndarray,
    h_m: float,
    h: float,
    us: np.ndarray,
    ys: np.ndarray,
    Ad: np.ndarray,
    Bd: np.ndarray,
    Cd: np.ndarray,
    Ed: np.ndarray,
    Qd: float,
    Rd: float,
) -> pd.DataFrame:
    """Example kalman filter for yaw and yaw rate

    Parameters
    ----------
    x0 : np.ndarray
        initial state [yaw, yaw rate]
    P_prd : np.ndarray
        2x2 array: initial covariance matrix
    h_m : float
        time step measurement [s]
    h : float
        time step filter [s]
    us : np.ndarray
        1D array: inputs
    ys : np.ndarray
        1D array: measured yaw
    Ad : np.ndarray
        2x2 array: discrete time transition matrix
    Bd : np.ndarray
        2x1 array: discrete time input transition matrix
    Cd : np.ndarray
        1x2 array: measurement transition matrix
    Ed : np.ndarray
        2x1 array
    Qd : float
        process noise
    Rd : float
        measurement noise

    Returns
    -------
    pd.DataFrame
        data frame with filtered data
    """
    x_prd = x0
    t = 0
    df = pd.DataFrame()

    for i in range(len(us)):

        u = us[i]  # input
        y = ys[i].T  # measurement

        for j in range(int(h_m / h)):
            t += h
            # Compute kalman gain:
            S = Cd @ P_prd @ Cd.T + Rd  # System uncertainty
            K = P_prd @ Cd.T @ inv(S)
            IKC = np.eye(2) - K @ Cd

            # State corrector:
            x_hat = x_prd + K * np.rad2deg(
                ssa(np.deg2rad(y - Cd @ x_prd))
            )  # smallest signed angle

            # corrector
            P_hat = IKC * P_prd @ IKC.T + K * Rd @ K.T

            # Predictor (k+1)
            x_prd = Ad @ x_hat + Bd * u
            P_prd = Ad @ P_hat @ Ad.T + Ed * Qd @ Ed.T

            s = pd.Series(name=t)
            s["yaw"] = x_hat[0][0]
            s["yaw rate"] = x_hat[1][0]
            s["yaw predictor"] = x_prd[0][0]
            s["yaw rate predictor"] = x_prd[1][0]

            df = df.append(s)

    return df


def extended_kalman_filter_example(
    x0: np.ndarray,
    P_prd: np.ndarray,
    lambda_f: Callable,
    lambda_jacobian: Callable,
    h_m: float,
    h: float,
    us: np.ndarray,
    ys: np.ndarray,
    Qd: float,
    Rd: float,
    a=-1,
    b=1,
    e=1,
) -> pd.DataFrame:
    """Example extended kalman filter

    Parameters
    ----------
    x0 : np.ndarray
        initial state [x_1, x_2]
    P_prd : np.ndarray
        2x2 array: initial covariance matrix
    h_m : float
        time step measurement [s]
    h : float
        time step filter [s]
    us : np.ndarray
        1D array: inputs
    ys : np.ndarray
        1D array: measured yaw
    Qd : float
        process noise
    Rd : float
        measurement noise

    Returns
    -------
    pd.DataFrame
        data frame with filtered data
    """
    x_prd = x0
    t = 0
    df = pd.DataFrame()
    E = np.array([[0, e]]).T
    no_states = len(P_prd)

    for i in range(len(us)):

        u = us[i]  # input
        y = ys[i].T  # measurement

        Cd = np.array([[1, 0]])  # Measurement!

        for j in range(int(h_m / h)):
            t += h
            # Compute kalman gain:
            S = Cd @ P_prd @ Cd.T + Rd  # System uncertainty
            K = P_prd @ Cd.T @ inv(S)
            IKC = np.eye(no_states) - K @ Cd

            # State corrector:
            P_hat = IKC * P_prd @ IKC.T + K * Rd @ K.T
            eps = y - Cd @ x_prd
            x_hat = x_prd + K * eps

            # discrete-time extended KF-model
            f_hat = lambda_f(a=a, b=b, u=u, w=0, x_2=float(x_hat[1]))

            f_d = x_hat + h * f_hat

            # Predictor (k+1)
            # Ad = I + h * A and Ed = h * E
            # where A = df/dx is linearized about x = x_hat
            Ad = lambda_jacobian(a=a, h=h, x_2=float(np.abs(x_hat[1])))

            Ed = h * E

            x_prd = f_d
            P_prd = Ad @ P_hat @ Ad.T + Ed * Qd @ Ed.T

            Cd = np.array([[0, 0]])  # No measurement!

            s = pd.Series(name=t)
            s["x_1"] = x_hat[0][0]
            s["x_2"] = x_hat[1][0]
            s["x_1 predictor"] = x_prd[0][0]
            s["x_2 predictor"] = x_prd[1][0]

            df = df.append(s)

    return df


def simulate_model(
    x0,
    us,
    ws,
    t,
    a=-1,
    b=1,
    e=1,
):

    simdata = [x0.flatten()]
    x = x0
    h = t[1] - t[0]
    for i, u in enumerate(us[0:-1]):

        u = us[i]  # input
        w = ws[i]  # process noise

        f = np.array([[float(x[1])], [float(a * x[1] * np.abs(x[1]) + b * u)]])
        E = np.array([[0, e]]).T
        x_dot = f + E * w

        ## Euler integration (k+1)
        x = x + h * x_dot

        simdata.append(x.flatten())

    simdata = np.array(simdata)
    df = pd.DataFrame(simdata, columns=["x_1", "x_2"], index=t)
    df["u"] = us
    df["w"] = ws
    return df
