"""Module for Vessel Manoeuvring Model (VMM) simulation

This is a variation where u refers to perturbed surge velocity
about nominal speed U0:
U = sqrt((U0 + u)^2 + v^2);

References:
[1] Matusiak, Jerzy. Dynamics of a Rigid Ship - with Applications, 3rd Edition, 2021.

[2] Chislett, M. S., and J. Strom-Tejsen. “Planar Motion Mechanis (PMM) Tests and Full Scale Steering and Manoeuvring Predictions for a Mariner Class Vessel.” Hydro- and Aerodynamics Laboratory, Hydrodynamics Section, Lyngby, Denmark, Report No. Hy-6, 1965. https://repository.tudelft.nl/islandora/object/uuid%3A6436e92f-2077-4be3-a647-3316d9f16ede.
"""
import pandas as pd
import numpy as np
from vessel_manoeuvring_models.models.vmm import Simulator, Result
from vessel_manoeuvring_models.substitute_dynamic_symbols import run
from scipy.spatial.transform import Rotation as R
from scipy.integrate import solve_ivp


class SimulatorPerturbed(Simulator):
    def step(
        self,
        t: float,
        states: np.ndarray,
        parameters: dict,
        ship_parameters: dict,
        control: pd.DataFrame,
        U0=1,
    ) -> np.ndarray:
        """Calculate states derivatives for next time step


        Parameters
        ----------
        t : float
            current time
        states : np.ndarray
            current states as a vector
        parameters : dict
            hydrodynamic derivatives
        ship_parameters : dict
            ship parameters lpp, beam, etc.
        control : pd.DataFrame
            data frame with time series for control devices such as rudder angle (delta) and popeller thrust.
        U0 : float
            initial velocity constant [1] (only used for linearized models)

        Returns
        -------
        np.ndarray
            states derivatives for next time step
        """

        u, v, r, x0, y0, psi = states

        states_dict = {
            "u": u,
            "v": v,
            "r": r,
            "x0": x0,
            "y0": y0,
            "psi": psi,
        }

        inputs = dict(parameters)
        inputs.update(ship_parameters)
        inputs.update(states_dict)

        if isinstance(control, pd.DataFrame):
            index = np.argmin(np.array(np.abs(control.index - t)))
            control_ = dict(control.iloc[index])
        else:
            control_ = control
        inputs.update(control_)

        inputs["U"] = U0  # initial velocity constant [1]
        inputs_perturbed = inputs.copy()
        inputs_perturbed["u"] -= (
            U0,
        )  # Note shis is the change in the perturbed version!!!
        inputs["X_qs"] = run(function=self.X_qs_lambda, **inputs_perturbed)
        inputs["Y_qs"] = run(function=self.Y_qs_lambda, **inputs_perturbed)
        inputs["N_qs"] = run(function=self.N_qs_lambda, **inputs_perturbed)

        u1d, v1d, r1d = run(function=self.acceleration_lambda, **inputs)

        # get rid of brackets:
        u1d = u1d[0]
        v1d = v1d[0]
        r1d = r1d[0]

        rotation = R.from_euler("z", psi, degrees=False)
        w = 0
        velocities = rotation.apply([u, v, w])
        x01d = velocities[0]
        y01d = velocities[1]
        psi1d = r
        dstates = [
            u1d,
            v1d,
            r1d,
            x01d,
            y01d,
            psi1d,
        ]
        return dstates
