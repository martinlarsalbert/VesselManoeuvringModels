import pandas as pd
import numpy as np


def to_test_matrix(
    df_VCT: pd.DataFrame, lpp: float, scale_factor: float = 1
) -> pd.DataFrame:
    """
    lpp : length of ship for df_VCT
    scale_factor : do you want to scale your matrix?
    """

    test_matrix = pd.DataFrame(index=df_VCT.index)
    test_matrix[["beta", "test_type", "r", "delta", "V"]] = df_VCT[
        ["beta", "test type", "r", "delta", "V"]
    ]
    test_matrix["beta"] = np.rad2deg(test_matrix["beta"])
    test_matrix["delta"] = np.rad2deg(test_matrix["delta"])
    test_matrix["jvf"] = 1

    test_matrix_save = test_matrix.copy()

    # No negative radious allowed...
    #df = test_matrix_save.groupby(by="test_type").get_group("Circle + Drift")
    #mask = (test_matrix_save["test_type"] == "Circle + Drift") & (
    #    test_matrix_save["r"] > 0
    #)
    #test_matrix_save.loc[mask, "r"] *= -1
    #test_matrix_save.loc[mask, "beta"] *= -1

    test_matrix_save["vs_kn"] = np.round(
        test_matrix_save["V"] * np.sqrt(scale_factor) * 3.6 / 1.852, decimals=5
    )
    test_matrix_save["radius"] = np.round(
        -test_matrix_save["V"] / test_matrix_save["r"] / lpp, decimals=5
    )  # Note minus!!!
    test_matrix_save["delta"] *= -1
    

    mask = np.isinf(test_matrix_save["radius"])
    test_matrix_save.loc[mask, "radius"] = np.abs(test_matrix_save.loc[mask, "radius"])

    test_matrix_save["test_type"] = test_matrix_save["test_type"].apply(
        lambda x: x.lower()
    )
    test_matrix_save["test_type"] = test_matrix_save["test_type"].apply(
        lambda x: x.replace(" + ", "_")
    )

    renames = {
        "circle_rudder angle": "circle_rudder",
        "rudder and drift angle": "drift_rudder",
        "drift angle": "drift",
        "rudder angle": "rudder",
        "thrust variation": "rudder_thrust",
    }
    for old, new in renames.items():
        mask = test_matrix_save["test_type"].str.contains(old)
        test_matrix_save.loc[mask, "test_type"] = new

    save_columns = [
        "beta",
        "test_type",
        #    "r",
        "delta",
        #    "V",
        "jvf",
        "vs_kn",
        "radius",
    ]

    return test_matrix_save[save_columns].copy()

def find_r(row, df_VCT:pd.DataFrame):

    mask = df_VCT['r_round'] == row['r_round']

    if mask.sum()==0:
        return row['r_round']
        
    return df_VCT.loc[mask].iloc[0]['r']