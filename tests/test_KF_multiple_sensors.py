from vessel_manoeuvring_models.KF_multiple_sensors import KalmanFilter
import numpy as np
import pytest
import pandas as pd

g = 9.81

@pytest.fixture
def kf():
    
    A = np.array([[0,1],
              [0,0]]) # Discrete time transition matrix
    B = np.array([[0],[1]])     # Discrete time input transition matrix

    H = np.array([[1,0]])        # Measurement transition matrix

    Q = np.array([
        [0, 0],
        [0, 0.1]
                  ]) # Covariance matrix of the process model

    var_x = 2
    R = np.array([[var_x**2]]) # Covariance matrix of the measurement
    
    kf = KalmanFilter(A=A,B=B, H=H, Q=Q, R=R, 
                      state_columns=['x','x1d'],
                      measurement_columns=['x'],
                      input_columns=['g'],
                      )
    
    yield kf
    

def test_predict(kf):
    
    dt = 0.1
    
    u = np.array([-g])
    
    x0 = np.array([[0,0]]).T
    P_0 = np.diag([0.1,0.01])
    u = np.array([[-g]]).T
    x_prd, P_prd = kf.predict(x_hat=x0, P_hat=P_0,u=u,h=dt)
    
def test_update(kf):
    
    dt = 0.1
    x0 = np.array([[0,0]]).T
    dt = 0.1
    t = np.arange(0,5,dt)
    
    u = np.array([[-g]]).T
    x_true = kf.simulate(x0=x0, t=t, us=u,)
    
@pytest.fixture
def data(kf):
    
    dt = 0.1
    x0 = np.array([[0,0]]).T
    dt = 0.1
    t = np.arange(0,5,dt)
    
    u = np.array([[-g]]).T
    x_true = kf.simulate(x0=x0, t=t, us=u,)
    
    np.random.seed(42)
    var_x = 2
    epsilon = np.random.normal(scale=var_x**2, size=len(t))
    ys=np.array([x_true[0,:]+epsilon])
    
    data = pd.DataFrame(index=t)
    data.index.name='time'
    
    data[kf.measurement_columns] = ys.T
    data['g'] = -g 
        
    yield data


def test_filter(kf, data):

    x0 = np.array([[0,0]]).T
    P_0 = np.diag([0.1,0.01])
    u = np.array([[-g]]).T
    dt = 0.1
    t = np.arange(0,5,dt)
    result = kf.filter(data=data, P_0=P_0, x0=x0)