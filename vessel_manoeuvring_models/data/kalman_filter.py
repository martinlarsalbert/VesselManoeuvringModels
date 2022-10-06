from pykalman import KalmanFilter
import numpy as np
import pandas as pd

def yaw(df:pd.DataFrame, observation_covariance=1000)->pd.DataFrame:
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

    df = df.copy()

    if not 'psi' in df:
        raise ValueError('df must contain heading "psi"')

    t = df.index.total_seconds()
    dt = t[1] - t[0]

    A = np.array([[1, dt, 0.5*(dt**2), 1/6*(dt**3)],
                  [0,  1, dt         , 0.5*(dt**2)],
                  [0,  0, 1          , dt],
                  [0,  0, 0          , 1]])

    kf = KalmanFilter(transition_matrices=A,
                      initial_state_mean = [df['psi'].iloc[0],0,0,0],
                      random_state=np.random.RandomState(0),
                      transition_covariance=100 * np.eye(4),
                      observation_covariance=observation_covariance * np.eye(1),

                      em_vars=[
                          'transition_covariance', 
                          #'observation_covariance', 
                          'initial_state_mean', 
                          'initial_state_covariance'
                          ],

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

def filter(df, key='x0', observation_covariance = 100000):
    
    df = df.copy()
    t = df.index.total_seconds()
    dt = t[1] - t[0]
    
    A = np.array([[1, dt, 0.5*(dt**2)],
                  [0,  1, dt         ],
                  [0,  0, 1          ],
                    ])
    
    kf = KalmanFilter(transition_matrices=A,
                      initial_state_mean = [df[key].iloc[0:100].median(),df[f'{key}1d_gradient'].iloc[0:100].median(),0],
                      random_state=np.random.RandomState(0),
                      transition_covariance=100 * np.eye(3),
                      observation_covariance=observation_covariance * np.eye(1),
    
                      em_vars=[
                          'transition_covariance', 
                          'observation_covariance', 
                          'initial_state_mean', 
                      #    'initial_state_covariance'
                          ],
    
                      )
    
    observations = df[key]
    states_pred = kf.em(observations).smooth(observations)[0]
    
    df[f'{key}_filtered'] = states_pred[:,0]
    df[f'{key}1d'] = states_pred[:,1]
    df[f'{key}2d'] = states_pred[:,2]
    
    return df


def filter_and_transform(df:pd.DataFrame, observation_covariance = 100000)->pd.DataFrame:

    df = df.copy()
    t = df.index.total_seconds()
    df['x01d_gradient'] = np.gradient(df['x0'], t)
    df['y01d_gradient'] = np.gradient(df['y0'], t)
    df['z01d_gradient'] = np.gradient(df['z0'], t)
    
    df['x02d_gradient'] = np.gradient(df['x01d_gradient'], t)
    df['y02d_gradient'] = np.gradient(df['y01d_gradient'], t)
    df['z02d_gradient'] = np.gradient(df['z01d_gradient'], t)

    df['psi1d_gradient'] = np.gradient(df['psi'], t)
    df['psi2d_gradient'] = np.gradient(df['psi1d_gradient'], t)

    df = filter(df=df, key='x0', observation_covariance = observation_covariance)
    df = filter(df=df, key='y0', observation_covariance = observation_covariance)
    df = filter(df=df, key='psi', observation_covariance = observation_covariance)
    df = transform_to_ship(df=df)
    return df

def transform_to_ship(df:pd.DataFrame, include_unfiltered=True)->pd.DataFrame:
    """transform to ship fixed velocities and accelerations

    Args:
        df : Dataframe with time derivatives: 
            velocities: x01d, y01d, z01d_gradient
            accelerations: x02d, y02d, z02d_gradient
            
    Returns:
        [pd.DataFrame]: new columns: u,v,r,u1d,v1d,w1d,r,r1d,beta
    """

    rotation = R.from_euler('z', df['psi_filtered'], degrees=False)
    df[['u','v','w']] = rotation.inv().apply(df[['x01d','y01d','z01d_gradient']])
    df[['u1d','v1d','w1d']] = rotation.inv().apply(df[['x02d','y02d','z02d_gradient']])

    df['r'] = df['psi1d']
    df['r1d'] = df['psi2d']
    df['beta'] = -np.arctan2(df['v'],df['u'])

    ## unfiltered
    if include_unfiltered:
        rotation = R.from_euler('z', df['psi'], degrees=False)
        df[['u_gradient','v_gradient','w_gradient']] = rotation.inv().apply(df[['x01d_gradient','y01d_gradient','z01d_gradient']])
        df[['u1d_gradient','v1d_gradient','w1d_gradient']] = rotation.inv().apply(df[['x02d_gradient','y02d_gradient','z02d_gradient']])

        df['r_gradient'] = df['psi1d_gradient']
        df['r1d_gradient'] = df['psi2d_gradient']


    return df