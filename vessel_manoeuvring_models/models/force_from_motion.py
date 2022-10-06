from vessel_manoeuvring_models.models.vmm import VMM
import pandas as pd
from vessel_manoeuvring_models.symbols import *
from vessel_manoeuvring_models.parameters import df_parameters
import sympy as sp
from vessel_manoeuvring_models.substitute_dynamic_symbols import run
from vessel_manoeuvring_models.prime_system import PrimeSystem

p = df_parameters["symbol"]


def predict_force(
    data: pd.DataFrame, added_masses: dict, ship_parameters: dict, vmm: VMM
) -> pd.DataFrame:
    """Predict forces and moment from motions using EOM and mass/added mass

    Parameters
    ----------
    data : pd.DataFrame
        Data to be regressed in SI units!
        That data should be a time series:
        index: time
        And the states:
        u,v,r,u1d,v1d,r1d
        and the inputs:
        delta,(thrust)

    added_masses : dict or pd.DataFrame with row: "prime"
        added masses in prime system units
    ship_parameters : dict
        ship parameters in SI units,
        ex:
        {
            "L": 100,        # Ship length [m]
            "rho": 1025,     # water density [kg/m3]
            "I_z": 100000000,# yaw mass moment of inertia [kg*m**2]
            "m": 10000000,   # mass of ship [kg]
            "x_G": 2.5,     # Longitudinal position of CG rel lpp/2 [m]
        }
    vmm : ModelSimulator
            vessel manoeuvring model
            either specified as:
            1) model simulator object
            2) or python module example: :func:`~vessel_manoeuvring_models.models.vmm_linear`

    Returns
    -------
    pd.DataFrame
        with fx,fy,mz predicted from motions added (SI units)
    """

    subs = [(value, key) for key, value in p.items()]
    subs.append((u1d, "u1d"))
    subs.append((v1d, "v1d"))
    subs.append((r1d, "r1d"))

    """
    Predicting the hydrodynamic forces (X_D) by inverting the EOM:
                                                  2            
    X_D =-X_{\dot{u}}⋅\dot{u} + \dot{u}⋅m - m⋅r ⋅x_G - m⋅r⋅v
    """
    solution = sp.solve(vmm.X_eq_separated, X_D)[0]
    solution = solution.subs(subs)
    lambda_X_D = sp.lambdify(list(solution.free_symbols), solution)

    solution = sp.solve(vmm.Y_eq_separated, Y_D)[0]
    solution = solution.subs(subs)
    lambda_Y_D = sp.lambdify(list(solution.free_symbols), solution)

    solution = sp.solve(vmm.N_eq_separated, N_D)[0]
    solution = solution.subs(subs)
    lambda_N_D = sp.lambdify(list(solution.free_symbols), solution)

    df = data.copy()
    df["U"] = np.sqrt(df["u"] ** 2 + df["v"] ** 2)

    ps = PrimeSystem(L=ship_parameters["L"], rho=ship_parameters["rho"])
    ship_parameters_prime = ps.prime(ship_parameters)

    columns = ["u", "v", "r", "u1d", "v1d", "r1d", "delta", "thrust", "U"]
    selection = list(set(columns) & set(df.columns))
    df_prime = ps.prime(df[selection], U=df["U"])

    df_prime["fx"] = run(
        lambda_X_D, inputs=df_prime, **ship_parameters_prime, **added_masses
    )
    df_prime["fy"] = run(
        lambda_Y_D, inputs=df_prime, **ship_parameters_prime, **added_masses
    )
    df_prime["mz"] = run(
        lambda_N_D, inputs=df_prime, **ship_parameters_prime, **added_masses
    )

    df_ = ps.unprime(df_prime, U=df["U"])
    df["fx"] = df_["fx"].values
    df["fy"] = df_["fy"].values
    df["mz"] = df_["mz"].values

    return df
