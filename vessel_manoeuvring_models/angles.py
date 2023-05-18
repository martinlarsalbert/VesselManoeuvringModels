import numpy as np


def smallest_signed_angle(angle: np.ndarray) -> np.ndarray:
    """smallest signed angle" or the smallest difference between two
    angles. Examples:
    angle = ssa(angle) maps an angle in rad to the interval [-pi pi)
        For feedback control systems and state estimators used to control the
    attitude of vehicles, the difference of two angles should always be
    mapped to [-pi pi) or [-180 180] to avoid step inputs/discontinuties.
    Author:     Thor I. Fossen
    Date:       2018-09-21
    _________________________________________________________________
    Parameters
    ----------
    angle : np.ndarray
        [rad]
    Returns
    -------
    np.ndarray
        smalles signed angle in [rad]
    """
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi


def mean_angle(angle: np.ndarray, axis=0) -> float:
    """Get mean angle

    Parameters
    ----------
    angle : np.ndarray
        Angle array [rad]

    Returns
    -------
    float
        mean angle [rad]
    """
    return np.arctan2(
        np.mean(np.sin(angle), axis=axis), np.mean(np.cos(angle), axis=axis)
    )
