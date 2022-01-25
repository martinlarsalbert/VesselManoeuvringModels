import numpy as np


def scale_force_to_model_scale(forces, rho, S, V, scale_factor, **kwargs):
    """
    Calculate force from equation using the equation:
    F = C*1/2*rho*V**2*S

    """

    def denominator(rho, V, S, **kwargs):
        return 1 / 2 * rho * V ** 2 * S

    S_m = S / (scale_factor ** 2)
    V_m = V / (np.sqrt(scale_factor))

    return forces.multiply(denominator(rho, V_m, S_m) / denominator(rho, V, S), axis=0)


def scale_moment_to_model_scale(forces, rho, S, V, lpp, scale_factor, **kwargs):
    """
    Calculate force from equation using the equation:
    F = C*1/2*rho*V**2*S*lpp

    """

    def denominator(rho, V, S, lpp, **kwargs):
        return 1 / 2 * rho * V ** 2 * S * lpp

    S_m = S / (scale_factor ** 2)
    V_m = V / (np.sqrt(scale_factor))
    lpp_m = lpp / scale_factor

    return forces.multiply(
        denominator(rho, V_m, S_m, lpp_m) / denominator(rho, V, S, lpp), axis=0
    )
