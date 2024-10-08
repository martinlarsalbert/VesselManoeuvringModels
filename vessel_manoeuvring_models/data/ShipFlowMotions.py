import pandas as pd
import numpy as np
from numpy import cos,sin,arctan2

from sympy import symbols, Eq
import sympy as sp
from vessel_manoeuvring_models.substitute_dynamic_symbols import lambdify
import os
from io import StringIO
from scipy.integrate import solve_ivp

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
    
    meta_data['R'] = R = meta_data['V']/meta_data['r']
    meta_data['t_ramp'] = meta_data['V']/acc_max
    meta_data['T'] = T = lambda_T(R=np.abs(meta_data['R']), V_max=meta_data['V'], t_ramp=meta_data['t_ramp']) 
    
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
    
    time_series['beta'] = meta_data['beta']
    time_series['u'] = time_series['V']*np.cos(time_series['beta'])
    time_series['v'] = -time_series['V']*np.sin(time_series['beta'])
    
    time_series['psi']+=time_series['beta']

    return time_series

def save_time_series(time_series, meta_data, ship_data, dir_name:str):

    columns=["x", "y", "z", "phi", "theta", "psi"]
    
    time_series_save = time_series[columns].copy()
    time_series_save['psi'] = np.rad2deg(time_series['psi'])

    s_data = time_series_save.to_csv(header=False, sep='\t')
    s_save=f"""txyzphithepsi
{len(time_series_save)-1}
{s_data}
""" 
    lpp = ship_data['L']*ship_data['scale_factor']
    
    if meta_data['r']!=0:
        R = meta_data['V']/meta_data['r']
        R_shipflow = -R/lpp
        
    
    if meta_data['test type'] == 'Circle':
        file_name = f"radius_{R_shipflow:.2f}.csv"
    elif meta_data['test type'] == 'Circle + Drift':
        file_name = f"radius_{R_shipflow:.2f}_beta{np.rad2deg(meta_data['beta']):.0f}.csv"
    elif meta_data['test type'] == 'pure yaw':
        V_kn = meta_data['V']*3.6/1.852
        file_name = f"V_{V_kn:.1f}_T_{meta_data['T']:.0f}s.csv"        
    else:
        file_name = f"{meta_data.name}.csv"

    V_kn = meta_data['V']*3.6/1.852
    dir_name_V = f"{V_kn:.0f}kn"
    
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    dir_path = os.path.join(dir_name,dir_name_V)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_path = os.path.join(dir_path, file_name)
    with open(file_path, mode='w') as file:
        file.write(s_save)
        
def test_load_time_series(file_path):

    with open(file_path, mode='r') as file:
        s_raw = file.readlines()
    
    s = "".join(s_raw[2:])

    df = pd.read_csv(StringIO(s), sep=r'\t', names=["time", "x", "y", "z", "phi", "theta", "psi"], engine='python')
    df.index = df['time']
    df.index.name ='time'
    
    df['psi'] = np.deg2rad(df['psi'])
    df['r'] = np.gradient(df['psi'],df.index)
    
    
    df['x1d'] = np.gradient(df['x'],df.index)
    df['y1d'] = np.gradient(df['y'],df.index)
    df['V'] = np.sqrt(df['x1d']**2 + df['y1d']**2)
     
    df['u'] = df['x1d']*np.cos(df['psi']) + df['y1d']*np.sin(df['psi'])
    df['v'] = -df['x1d']*np.sin(df['psi']) + df['y1d']*np.cos(df['psi'])
    df['beta'] = -np.arctan2(df['v'],df['u'])
        

    return df

def read_MOTIONS(file_path:str, do_mirror=True)-> pd.DataFrame:
    """_summary_

    Args:
        file_path (str): _description_
        do_mirror (bool, optional): Z axis is upward in MOTIONS, so mirroring is needed. Defaults to True.

    Returns:
        pd.DataFrame: _description_
    """
    
    if isinstance(file_path, str):
        df = pd.read_csv(file_path)
        df.index = df['Time']
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

def steady_state(df, tol = 10**-7, tight=False):
    
    i_start = (df['V'].diff().abs() < tol).idxmax()
    
    if tight:
        df_steady = df.loc[i_start:].copy()
        return df_steady

        
    i_steady = (df.index[-1] - i_start)/2 + i_start

    df_steady = df.loc[i_steady:].copy()
    
    
    mask = (df_steady['V'].diff().abs() > tol)
    if mask.sum()>0:
        i_end = mask.idxmax()
        
        i_steady = (i_end - df_steady.index[0])*5/6 + df_steady.index[0]
        
        df_steady = df_steady.loc[0:i_steady].copy()
    

    return df_steady  

## Pure yaw
def step(t,x, psi_max, u_max, T, t_ramp=100, acc_max=0.04):
    
    x0,y0,psi,u,v,r = x

    if u < u_max:
        u1d=acc_max
    else:
        u1d=0
        u=u_max
    
    x01d = u * np.cos(psi) - v * np.sin(psi)
    y01d = u * np.sin(psi) + v * np.cos(psi)

    if t < t_ramp:
        psi_a = psi_max/t_ramp*t
    else:
        psi_a = psi_max

    w = 2*np.pi/T    
    r1d = -psi_a*w**2*np.sin(w*t)
        
    dx = [x01d,y01d,r,u1d,0,r1d]

    return dx

def create_pure_yaw(psi_max, u_max, T, t_ramp=100, acc_max=0.04,t_max=1000,dt=0.1, y0 = [0,0,0,0,0,0], max_step:float=None,**kwargs):

    t_eval = np.arange(0,t_max,dt)
    t_span = (t_eval[0], t_eval[-1])
    
    #u0 = 0
    #r0 = 0
    #y0 = [0,0,0,u0,0,r0]
    
    if max_step is None:
        max_step = dt
    
    result = solve_ivp(fun=step, t_span=t_span, y0=y0, t_eval=t_eval, args=(psi_max,u_max,T,t_ramp,acc_max), max_step=max_step, **kwargs)

    time_series = pd.DataFrame(result.y.T, index = result.t, columns=['x','y','psi','u','v','r'])
    time_series[['z', 'phi', 'theta']]=0
    

    meta_data = {
    'V':u_max,
    'psi_max':psi_max,
    'u_max':u_max,
    'r':0,
    'test type':'pure yaw',
    'T':T,
    }

    return time_series, meta_data 