import numpy as np
import pandas as pd


def variate(df: pd.DataFrame, variation_keys: list, N: int = 10) -> pd.DataFrame:
    """Linearly spaced variation of values in a data frame.
    This can be used to simulate a captive test by sampling a mathematical model.
    Ex:
    df = u:[1.5..2.5], delta:[0..35], v:[0..0.5]
    variation_keys = ['u','delta']
    --> All possible combinations between the min/max of u and delta are varied at the same time.


    Parameters
    ----------
    df : pd.DataFrame
        [description]
    variation_keys : list
        [description]
    N : int, optional
        [description], by default 10

    Returns
    -------
    pd.DataFrame
        All possible variations
        Giving a dataframe with length: N**(len(variation_keys))
    """
    variations = []
    for variation_key in variation_keys:
        variation = np.linspace(df[variation_key].min(), df[variation_key].max(), N)
        variations.append(variation)

    matrix = np.meshgrid(*variations)
    df_variation = pd.DataFrame()
    for variation_key, values in zip(variation_keys, matrix):
        df_variation[variation_key] = values.flatten()

    return df_variation
