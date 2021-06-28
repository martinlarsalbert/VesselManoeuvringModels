import os
import pandas as pd
from mdl_helpers import mdl_motions
from typing import Union

def runs()->pd.DataFrame:
    """Get meta data about the runs

    Returns:
        pd.DataFrame: Meta data for all runs
    """
    df_runs = pd.read_csv('../data/raw/runs.csv', index_col=0)
    return df_runs

def load_run(id:int, dir_path='../data/raw')->pd.DataFrame:
    """Load time series for one run.

    Args:
        id (int): id of run

    Returns:
        pd.DataFrame: time series as a data frame.
    """
    
    file_name = f'{id}.csv'
    file_path = os.path.join(dir_path, file_name)
    df = pd.read_csv(file_path, index_col=0)
    
    df.index = pd.TimedeltaIndex(df.index,unit='s')
    
    return df

def load_units()->pd.Series:
    """load units of the time series

    Returns:
        pd.Series: units
    """
    units = pd.read_csv('../data/raw/units.csv', index_col=0)['0']
    return units

def load_meta_data(id:int)->pd.Series:
    """Load run meta data (run number, model data etc.)

    Args:
        id (int): id of run

    Returns:
        pd.Series: run meta data
    """

    df_runs = runs()
    meta_data = df_runs.loc[id]
    return meta_data

def preprocess_run(df:pd.DataFrame, meta_data:pd.Series, units:pd.Series):
    
    meta_data = {
        'ScaleFactor' : meta_data.scale_factor,
        'LOG' : meta_data.lcg,
        'KG' : meta_data.kg,
        'xm' : meta_data.xm,
        'ym' : meta_data.ym,
        'zm' : meta_data.zm,
    }
    meta_data = pd.Series(meta_data)

    time_df, units = mdl_motions.add_ModelPos_motions(df=df, units = units, meta_data=meta_data)
    time_df['x0']-=time_df['x0'].iloc[0]
    time_df['y0']-=time_df['y0'].iloc[0]
    time_df['z0']-=time_df['z0'].iloc[0]
    time_df['psi']-=time_df['psi'].iloc[0]

    renames = {
        r'Rudder/Angle':'delta',
    }
    time_df.rename(columns=renames, inplace=True)
    for old,new in renames.items():
        units[new] =units[old]

    return time_df,units

def load(id:int, dir_path='../data/raw')->Union[pd.DataFrame,dict,pd.Series]:
    """[summary]

    Args:
        id (int): [description]

    Returns:
        Union[pd.DataFrame,dict,pd.Series]: [description]
    """

    df = load_run(id=id, dir_path=dir_path)
    units = load_units()
    meta_data = load_meta_data(id=id)
    
    return df,units, meta_data

    