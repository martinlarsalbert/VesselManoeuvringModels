import numpy as np
import pandas as pd
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
from scipy.interpolate import interp1d


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

    time = t_[0]

    if not rev is None:
        df_["rev"] = interpolated_control(t=df_.index, item=rev)
    if not twa is None:
        df_["twa"] = interpolated_control(t=df_.index, item=twa)
    if not tws is None:
        df_["tws"] = interpolated_control(t=df_.index, item=tws)

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
    )
    result["delta"] = df_["delta"]
    result["rev"] = df_["rev"]
    result["twa"] = df_["twa"]
    result["tws"] = df_["tws"]
    df_result = pd.concat((df_result, result), axis=0)
    time = df_result.index[-1]

    ## 2)
    direction *= -1

    t_ = np.arange(time, time + t_max, dt)
    data = np.tile(df_result.iloc[-1], (len(t_), 1))
    df_ = pd.DataFrame(data=data, columns=df_result.columns, index=t_)
    if not rev is None:
        df_["rev"] = interpolated_control(t=df_.index, item=rev)
    if not twa is None:
        df_["twa"] = interpolated_control(t=df_.index, item=twa)
    if not tws is None:
        df_["tws"] = interpolated_control(t=df_.index, item=tws)

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
    )
    result["delta"] = df_["delta"]
    result["rev"] = df_["rev"]
    result["twa"] = df_["twa"]
    result["tws"] = df_["tws"]
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
        df_["rev"] = interpolated_control(t=df_.index, item=rev)
    if not twa is None:
        df_["twa"] = interpolated_control(t=df_.index, item=twa)
    if not tws is None:
        df_["tws"] = interpolated_control(t=df_.index, item=tws)

    result = model.simulate(
        df_=df_,
        method=method,
        name=name,
        additional_events=additional_events,
    )
    result["delta"] = df_["delta"]
    result["rev"] = df_["rev"]
    result["twa"] = df_["twa"]
    result["tws"] = df_["tws"]

    df_result = pd.concat((df_result, result.iloc[1:]), axis=0)
    time = df_result.index[-1]

    df_result["V"] = np.sqrt(df_result["u"] ** 2 + df_result["v"] ** 2)

    return df_result


def interpolated_control(t, item: pd.Series):
    if isinstance(item, pd.Series):
        f = f = interp1d(
            x=item.index,
            y=item,
            kind="nearest",
            bounds_error=False,
            fill_value="extrapolate",
        )
        interpolated_item = f(t)
    elif np.isscalar(item):
        interpolated_item = item
    else:
        raise ValueError("Cannot interpolate control")
    return interpolated_item

def turning_circle(
    model,
    u0: float,
    rev: float = None,
    twa: float = None,
    tws: float = None,
    angle: float = 35.0,
    neutral_rudder_angle: float = 0.0,
    t_max: float = 1000.0,
    dt: float = 0.01,
    rudder_rate=2.32,
    method="Radau",
    name="simulation",
    r_limit = 0.1,
    **kwargs,
) -> pd.DataFrame:
    """Turning circle simulation

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
        Rudder angle [deg], by default 35.0 [deg]
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

    time = t_[0]

    if not rev is None:
        df_["rev"] = interpolated_control(t=df_.index, item=rev)
    if not twa is None:
        df_["twa"] = interpolated_control(t=df_.index, item=twa)
    if not tws is None:
        df_["tws"] = interpolated_control(t=df_.index, item=tws)

    direction = np.sign(angle)
    def course_deviated(t, states, control):
        u, v, r, x0, y0, psi = states
        target_psi = -direction * np.deg2rad(360)
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
    )
    result["delta"] = df_["delta"]
    result["rev"] = df_["rev"]
    result["twa"] = df_["twa"]
    result["tws"] = df_["tws"]
    df_result = pd.concat((df_result, result), axis=0)
    time = df_result.index[-1]

    ## 2)
    direction *= -1

    t_ = np.arange(time, time + t_max, dt)
    data = np.tile(df_result.iloc[-1], (len(t_), 1))
    df_ = pd.DataFrame(data=data, columns=df_result.columns, index=t_)
    if not rev is None:
        df_["rev"] = interpolated_control(t=df_.index, item=rev)
    if not twa is None:
        df_["twa"] = interpolated_control(t=df_.index, item=twa)
    if not tws is None:
        df_["tws"] = interpolated_control(t=df_.index, item=tws)

    # t_local = np.arange(0, t_max, dt)
    t_local = t_[t_ >= time] - time
    delta_ = np.deg2rad(angle) + direction * np.deg2rad(rudder_rate) * t_local
    mask = np.abs(delta_) > 0
    delta_[mask] = 0
    df_["delta"] = delta_ + np.deg2rad(neutral_rudder_angle)

    def r_stable(t, states, control):
        u, v, r, x0, y0, psi = states
        remain = r_limit - np.abs(r)
        return remain

    r_stable.terminal = True

    additional_events = [
        r_stable,
    ]
    
    result = model.simulate(
        df_=df_,
        method=method,
        name=name,
        additional_events=additional_events,
    )
    result["delta"] = df_["delta"]
    result["rev"] = df_["rev"]
    result["twa"] = df_["twa"]
    result["tws"] = df_["tws"]
    df_result = pd.concat((df_result, result.iloc[1:]), axis=0)
    time = df_result.index[-1]
    
    df_result["V"] = np.sqrt(df_result["u"] ** 2 + df_result["v"] ** 2)

    return df_result