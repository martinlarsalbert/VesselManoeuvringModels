import numpy as np
from numpy import sqrt, sin, cos, arctan2, pi
import sympy as sp
from sympy.vector import CoordSys3D, express
from vessel_manoeuvring_models.symbols import *

## Apparent wind equation
tws, twa, cog = sp.symbols("tws twa cog")
N = CoordSys3D("N")
S = N.orient_new_axis("S", psi, N.k)

W = -tws * sp.cos(twa) * N.i + -tws * sp.sin(twa) * N.j
V = U * sp.cos(cog) * N.i + U * sp.sin(cog) * N.j
H = -V
A = W + H

## aws and awa:
A_s = express(A, S)
eq_aws = sp.Eq(aws, sp.simplify(A_s.magnitude()))
eq_awa = sp.Eq(awa, sp.simplify(sp.atan2(A_s.dot(S.j), A_s.dot(S.i)) + sp.pi))

## cog:
V_S = u * S.i + v * S.j
V_N = express(V_S, N)
eq_cog = sp.Eq(cog, sp.atan2(V_N.dot(N.j), (V_N.dot(N.i))))


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
