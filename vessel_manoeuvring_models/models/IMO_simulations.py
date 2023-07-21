import numpy as np
import pandas as pd
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator


def zigzag(
    model,
    u0: float,
    rev: float = None,
    twa: float = None,
    tws: float = None,
    angle: float = 10.0,
    heading_deviation: float = 10.0,
    neutral_rudder_angle: float = 0.0,
    t_max: float = 1000.0,
    dt: float = 0.01,
    rudder_rate=2.32,
    method="Radau",
    name="simulation",
    **kwargs,
) -> pd.DataFrame:
    """ZigZag simulation

    Parameters
    ----------
    u0 : float
        initial speed [m/s]
    rev : float
        propeller speed [1/s]
    twa : float
        true wind angle [rad]
    tws : float
        true wind speed [m/s]
    angle : float, optional
        Rudder angle [deg], by default 10.0 [deg]
    heading_deviation : float
        normally same as angle [deg] --> zigzag10/10 etc. Otherwise 10/5 or similar.
    neutral_rudder_angle : float, by default 0.0
        neutral rudder angle [deg] so that actual rudder angle delta_real = delta + neutral_rudder_angle
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
    if not twa is None:
        df_["twa"] = twa
    if not tws is None:
        df_["tws"] = tws

    zig_zag_angle = np.abs(heading_deviation) + np.abs(
        neutral_rudder_angle
    )  # Note nautral rudder angle here!
    direction = np.sign(angle)

    def course_deviated(t, states, control):
        u, v, r, x0, y0, psi = states
        target_psi = -direction * np.deg2rad(zig_zag_angle)
        remain = psi - target_psi
        return remain

    course_deviated.terminal = True

    additional_events = [
        course_deviated,
    ]

    df_result = pd.DataFrame()

    ## 1)

    t_local = np.arange(0, t_max, dt)
    delta_ = direction * np.deg2rad(rudder_rate) * t_local
    mask = np.abs(delta_) > np.deg2rad(np.abs(angle))
    delta_[mask] = direction * np.abs(np.deg2rad(angle))
    df_["delta"] = delta_ + np.deg2rad(neutral_rudder_angle)

    result = model.simulate(
        df_=df_,
        method=method,
        name=name,
        additional_events=additional_events,
        include_accelerations=False,
        **kwargs,
    )
    result["delta"] = df_["delta"]
    df_result = pd.concat((df_result, result), axis=0)
    time = df_result.index[-1]

    ## 2)
    direction *= -1

    t_ = np.arange(time, time + t_max, dt)
    data = np.tile(df_result.iloc[-1], (len(t_), 1))
    df_ = pd.DataFrame(data=data, columns=df_result.columns, index=t_)
    if not rev is None:
        df_["rev"] = rev
    if not twa is None:
        df_["twa"] = twa
    if not tws is None:
        df_["tws"] = tws

    # t_local = np.arange(0, t_max, dt)
    t_local = t_[t_ >= time] - time
    delta_ = np.deg2rad(angle) + direction * np.deg2rad(rudder_rate) * t_local
    mask = np.abs(delta_) > np.deg2rad(np.abs(angle))
    delta_[mask] = direction * np.abs(np.deg2rad(angle))
    df_["delta"] = delta_ + np.deg2rad(neutral_rudder_angle)

    #
    result = model.simulate(
        df_=df_,
        method=method,
        name=name,
        additional_events=additional_events,
        include_accelerations=False,
        **kwargs,
    )
    result["delta"] = df_["delta"]
    df_result = pd.concat((df_result, result.iloc[1:]), axis=0)
    time = df_result.index[-1]

    ## 3)
    direction *= -1

    t_ = np.arange(time, time + t_max, dt)
    data = np.tile(df_result.iloc[-1], (len(t_), 1))
    df_ = pd.DataFrame(data=data, columns=df_result.columns, index=t_)

    # t_local = np.arange(0, t_max, dt)
    t_local = t_[t_ >= time] - time
    delta_ = (
        # -direction * np.deg2rad(angle) + direction * np.deg2rad(rudder_rate) * t_local
        -np.deg2rad(angle)
        + direction * np.deg2rad(rudder_rate) * t_local
    )
    mask = np.abs(delta_) > np.deg2rad(np.abs(angle))
    delta_[mask] = direction * np.abs(np.deg2rad(angle))
    df_["delta"] = delta_ + np.deg2rad(neutral_rudder_angle)
    if not rev is None:
        df_["rev"] = rev
    if not twa is None:
        df_["twa"] = twa
    if not tws is None:
        df_["tws"] = tws

    result = model.simulate(
        df_=df_,
        method=method,
        name=name,
        additional_events=additional_events,
        include_accelerations=False,
        **kwargs,
    )
    result["delta"] = df_["delta"]
    df_result = pd.concat((df_result, result.iloc[1:]), axis=0)
    time = df_result.index[-1]

    df_result["V"] = np.sqrt(df_result["u"] ** 2 + df_result["v"] ** 2)

    return df_result
