import pandas as pd
import numpy as np
from numpy import cos,sin,arctan2

from sympy import symbols, Eq
import sympy as sp
from vessel_manoeuvring_models.substitute_dynamic_symbols import lambdify

S, V, psi, t, t_ramp, V_max, acc, T, R = symbols("S, V, psi, t, t_ramp, V_max, acc, T, R")
eq_acc = Eq(acc,sp.Piecewise((V_max/t_ramp,t<=t_ramp),(0,t>t_ramp)))
eq_V = sp.Eq(V,sp.Integral(eq_acc.rhs, t).doit())
eq_S = sp.Eq(S,sp.Integral(eq_V.rhs, t).doit())
eq_T = sp.Eq(2*sp.pi*R,eq_S.rhs.args[1].args[0])
eq_T = Eq(T,sp.solve(eq_T,t)[0])
eq_T = Eq(T,sp.solve(eq_T,T)[0])
eq_psi = Eq(psi,eq_S.rhs/(2*sp.pi*R)*2*sp.pi)

lambda_V = lambdify(eq_V.rhs)
lambda_S = lambdify(eq_S.rhs)
lambda_T = lambdify(eq_T.rhs)
lambda_psi = lambdify(eq_psi.rhs)

def create_time_series(meta_data, fz = 0.1, N:int=None, acc_max=0.02):

    meta_data = meta_data.copy()
    
    meta_data['R'] = R = np.abs(meta_data['V']/meta_data['r'])
    meta_data['t_ramp'] = meta_data['V']/acc_max
    meta_data['T'] = T = lambda_T(R=meta_data['R'], V_max=meta_data['V'], t_ramp=meta_data['t_ramp']) 
    
    if N is None:
        N = int(np.ceil(T/fz))
    
    ts = np.linspace(0,T,N)
    
    time_series = pd.DataFrame(index=ts)
    time_series.index.name='time'
    columns=["x", "y", "z", "phi", "theta", "psi"]
    time_series[columns] = 0.0
    
    time_series['V'] = lambda_V(V_max=meta_data['V'], t=ts, t_ramp=meta_data['t_ramp'])
    time_series['S'] = lambda_S(V_max=meta_data['V'], t=ts, t_ramp=meta_data['t_ramp'])
    time_series['psi'] = psi = lambda_psi(R=meta_data['R'], V_max=meta_data['V'], t=ts, t_ramp=meta_data['t_ramp'])
    
    time_series['y'] = -R*np.cos(-psi) + R
    time_series['x'] = -R*np.sin(-psi)

    return time_series

def read_MOTIONS(file_path:str, do_mirror=True)-> pd.DataFrame:
    """_summary_

    Args:
        file_path (str): _description_
        do_mirror (bool, optional): Z axis is upward in MOTIONS, so mirroring is needed. Defaults to True.

    Returns:
        pd.DataFrame: _description_
    """
    
    if isinstance(file_path, str):
        df = pd.read_csv(file_path, index_col=0)
    elif isinstance(file_path, pd.DataFrame):
        df = file_path
    else:
        raise ValueError(f"Bad datatype for file_path ({type(file_path)})")
    
    
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

def steady_state(df, tol = 10**-7):
    
    i_start = (df['V'].diff().abs() < tol).idxmax()
        
    i_steady = (df.index[-1] - i_start)/2 + i_start

    df_steady = df.loc[i_steady:].copy()

    mask = (df_steady['V'].diff().abs() > tol)
    if mask.sum()>0:
        i_end = mask.idxmax()
        
        i_steady = (i_end - df_steady.index[0])*5/6 + df_steady.index[0]
        
        df_steady = df_steady.loc[0:i_steady].copy()
    

    return df_steady  