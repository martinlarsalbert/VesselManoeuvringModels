from pykalman import KalmanFilter
import numpy as np
import pandas as pd

def yaw(df:pd.DataFrame)->pd.DataFrame:
    """Kalman filter for yaw motion

    Args:
        df (pd.DataFrame): time in [s] as index

    Returns:
        pd.DataFrame: df with new columns:
            psi_filtered    : filtered heading
            r               : yaw rate [rad/s]
            r1d             : yaw rate acceleration [rad/s2]
            r2d             : yaw rate jerk [rad/s3]

    """

    if not 'psi' in df:
        raise ValueError('df must contain heading "psi"')

    t = df.index.total_seconds()
    dt = t[1] - t[0]

    A = np.array([[1, dt, 0.5*(dt**2), 1/6*(dt**3)],
                  [0,  1, dt         , 0.5*(dt**2)],
                  [0,  0, 1          , dt],
                  [0,  0, 0          , 1]])

    kf = KalmanFilter(transition_matrices=A,
                      initial_state_mean = [df['psi'].mean(),0,0,0],
                      random_state=np.random.RandomState(0),
                      #transition_covariance=100 * np.eye(3),
                      #observation_covariance=100000 * np.eye(1),
                      #em_vars=[
                      #    'transition_covariance', 
                      #    #'observation_covariance', 
                      #    'initial_state_mean', 
                      #    'initial_state_covariance'
                      #    ],
                      )

    observations = df['psi']
    states_pred = kf.em(observations).smooth(observations)[0]
    
    df['psi_filtered'] = states_pred[:,0]
    df['r'] = states_pred[:,1]
    df['r1d'] = states_pred[:,2]
    df['r2d'] = states_pred[:,3]

    return df