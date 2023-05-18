import numpy as np
from numpy import sqrt, sin, cos, arctan2, pi


def apparent_wind_speed_to_true(
    U: np.ndarray,
    awa: np.ndarray,
    aws: np.ndarray,
    cog: np.ndarray,
    psi: np.ndarray,
    **kwargs,
) -> np.ndarray:
    return sqrt(U**2 - 2 * U * aws * cos(awa - cog + psi) + aws**2)


def apparent_wind_angle_to_true(
    U: np.ndarray,
    awa: np.ndarray,
    aws: np.ndarray,
    cog: np.ndarray,
    psi: np.ndarray,
    **kwargs,
) -> np.ndarray:
    return (
        arctan2(
            U * sin(cog) - aws * sin(awa + psi), U * cos(cog) - aws * cos(awa + psi)
        )
        + pi
    )


def true_wind_speed_to_apparent(
    U: np.ndarray, cog: np.ndarray, twa: np.ndarray, tws: np.ndarray, **kwargs
) -> np.ndarray:
    return sqrt(U**2 + 2 * U * tws * cos(cog - twa) + tws**2)


def true_wind_angle_to_apparent(
    U: np.ndarray,
    cog: np.ndarray,
    psi: np.ndarray,
    twa: np.ndarray,
    tws: np.ndarray,
    **kwargs,
) -> np.ndarray:
    return (
        arctan2(
            -U * sin(cog - psi) + tws * sin(psi - twa),
            -U * cos(cog - psi) - tws * cos(psi - twa),
        )
        + pi
    )
