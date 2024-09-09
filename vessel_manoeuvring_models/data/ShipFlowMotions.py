import pandas as pd
import numpy as np
from numpy import cos,sin,arctan2

def read_MOTIONS(file_path:str, do_mirror=True)-> pd.DataFrame:
    """_summary_

    Args:
        file_path (str): _description_
        do_mirror (bool, optional): Z axis is upward in MOTIONS, so mirroring is needed. Defaults to True.

    Returns:
        pd.DataFrame: _description_
    """
    
    df = pd.read_csv(file_path, index_col=0)
    if do_mirror:
        df = mirror(df)
    
    df['x0'] = df['P1']
    df['y0'] = df['P2']
    df['z0'] = df['P3']
    df['phi'] = np.deg2rad(df['P4'])
    df['theta'] = np.deg2rad(df['P5'])
    df['psi'] = psi = np.deg2rad(df['P6'])
    
    
    df['u'] = df['V1']*cos(psi) + df['V2']*sin(psi)
    df['v'] = -df['V1']*sin(psi) + df['V2']*cos(psi)
    df['V'] = np.sqrt(df['u']**2+df['v']**2)
    df['beta'] = np.arctan2(-df['v'],df['u'])
    
    df['p'] = np.deg2rad(df['V4'])
    df['q'] = np.deg2rad(df['V5'])
    df['r'] = np.deg2rad(df['V6'])
       
    
    df['X_D'] = df['FIBD1']
    df['Y_D'] = df['FIBD2']
    df['N_D'] = df['FIBD6']
    
    return df

def mirror(df)->pd.DataFrame:
    
    mirror_dofs = ['2','4','6']
    
    columns_mirror = []
    
    for dof in mirror_dofs:
        columns_mirror+=[column for column in df.columns if dof in column]
        
    df_mirror = df.copy()
    df_mirror[columns_mirror]*=-1
    
    return df_mirror

def steady_state(df, tol = 10**-9):
    
    i_start = (df['V'].diff().abs() < tol).idxmax()
    i_steady = (df.index[-1] - i_start)/2 + i_start

    df_steady = df.loc[i_steady:].copy()

    return df_steady  