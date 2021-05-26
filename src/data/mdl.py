import os
import pandas as pd

def runs()->pd.DataFrame:
    """Get meta data about the runs

    Returns:
        pd.DataFrame: Meta data for all runs
    """
    df_runs = pd.read_csv('../data/raw/runs.csv', index_col=0)
    return df_runs

def load_run(id:int)->pd.DataFrame:
    """Load time series for one run.

    Args:
        id (int): id of run

    Returns:
        pd.DataFrame: time series as a data frame.
    """
    
    file_name = f'{id}.csv'
    file_path = os.path.join('../data/raw', file_name)
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