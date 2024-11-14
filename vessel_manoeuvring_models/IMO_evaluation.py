import numpy as np
import pandas as pd

def maximums(data, angle=10):

    start = (data['psi'].abs() > np.deg2rad(abs(angle))).idxmax()
    data = data.loc[start:]
    r_ = data['r'].values
    mask = (r_[0:-1] > 0) & (r_[1:] <= 0)
    mask = np.concatenate((mask,[False]))
    df_maximums = data.loc[mask]
    return df_maximums

def minimums(data, angle=10):
    start = (data['psi'].abs() > np.deg2rad(abs(angle))).idxmax()
    data = data.loc[start:]
    
    r_ = data['r'].values
    mask = (r_[0:-1] < 0) & (r_[1:] >= 0)
    mask = np.concatenate((mask,[False]))
    df_minimums = data.loc[mask]
    return df_minimums

def measure_overshoots(data, angle=10):

    df_maximums = maximums(data=data, angle=angle)
    df_minimums = minimums(data=data, angle=angle)
    
    peaks = pd.concat((df_maximums, df_minimums), axis=0).sort_index()
    overshoots = np.rad2deg(peaks['psi'].abs() - data['delta'].abs().max())
    return overshoots