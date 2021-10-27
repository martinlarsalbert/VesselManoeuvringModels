import pandas as pd
import numpy as np
import sympy as sp
from src.substitute_dynamic_symbols import lambdify, run
from src import symbols

from src.parameters import Xudot_, df_parameters
from src.models.diff_eq_to_matrix import DiffEqToMatrix
from src.symbols import *
import src.models.vmm as vmm
import statsmodels.api as sm


def results_summary_to_dataframe(results):
    """take the result of an statsmodel results
    table and transforms it into a dataframe"""
    pvals = results.pvalues
    coeff = results.params
    conf_lower = results.conf_int()[0]
    conf_higher = results.conf_int()[1]

    results_df = pd.DataFrame(
        {
            "$P_{value}$": pvals,
            "coeff": coeff,
            "$conf_{lower}$": conf_lower,
            "$conf_{higher}$": conf_higher,
        }
    )

    # Reordering...
    results_df = results_df[
        ["coeff", "$P_{value}$", "$conf_{lower}$", "$conf_{higher}$"]
    ]
    return results_df


def model_from_forces(model: vmm.Simulator, data: pd.DataFrame):
    N_ = sp.symbols("N_")

    diff_eq_N = DiffEqToMatrix(ode=model.N_qs_eq.subs(N_qs,N_),
    label=N_,base_features=[delta,u,v,r])

    X=diff_eq_N.calculate_features(data=data)
    y=diff_eq_N.calculate_label(y=data['mz'])
    model_N=sm.OLS(y,X)
    results_N = model_N.fit()
