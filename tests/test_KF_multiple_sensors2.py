from vessel_manoeuvring_models.KF_multiple_sensors import KalmanFilter
import numpy as np
import pytest
import pandas as pd

g = 9.81

@pytest.fixture
def kf():
    
    A = np.array([
        [0,1, 0 ],
        [0,0,-1],
        [0,0, 0 ],             
             ]) # Discrete time transition matrix

    B = np.array([[]])     # Discrete time input transition matrix # Discrete time transition matrix
    
    
    var_x = 2
    var_x1d = 0.001

    var_x_Q = 10
    var_x1d_Q = 10
    Q = np.diag([var_x_Q**2,var_x1d_Q**2,0]) # Covariance matrix of the process model
    
    H = np.array([
    [1,0,0],
    [0,1,0],
               ])        # Measurement transition matrix

    R = np.diag([var_x**2,var_x1d**2/10]) # Covariance matrix of the measurement
    
    kf = KalmanFilter(A=A,B=B, H=H, Q=Q, R=R, 
                      state_columns=['x','x1d','g'],
                      measurement_columns=['x','x1d'],
                      input_columns=[],
                      )
    
    yield kf
    

def test_predict(kf):
    
    dt = 0.1
    
    u = np.array([[]])
    x0 = np.array([[0,0,g]]).T
    P_0 = np.diag([0.1,0.01,0])
    u = np.array([[-g]]).T
    x_prd, P_prd = kf.predict(x_hat=x0, P_hat=P_0,u=u,h=dt)
    
def test_update(kf):
    
    dt = 0.1
    x0 = np.array([[0,0,g]]).T
    dt = 0.1
    t = np.arange(0,5,dt)
    
    u = np.array([[]])
    x_true = kf.simulate(x0=x0, t=t, us=u,)
    
@pytest.fixture
def ys(kf):
    
    dt = 0.1
    x0 = np.array([[0,0,g]]).T

    t = np.arange(0,5,dt)
    
    u = np.array([[]])
    x_true = kf.simulate(x0=x0, t=t, us=u,)
    
    np.random.seed(42)
    var_x = 2
    epsilon = np.random.normal(scale=var_x**2, size=len(t))
    ys=x_true.copy()
    ys[0,:]+=epsilon
    ys = np.array(ys[0:2,:])
    
    yield ys


#def test_filter(kf, ys):
#
#    x0 = np.array([[0,0,g]]).T
#    P_0 = np.diag([0.1,0.01,0])
#    u = np.array([[]])
#    dt = 0.1
#    t = np.arange(0,5,dt)
#    
#    result = kf.filter(x0=x0, P_0=P_0, us=u, ys=ys, t=t)
    
@pytest.fixture
def data(kf):
    
    dt = 0.1
    x0 = np.array([[0,0,g]]).T
    dt = 0.1
    t = np.arange(0,5,dt)
    
    u = np.array([[]])
    x_true = kf.simulate(x0=x0, t=t, us=u,)
    
    np.random.seed(42)
    var_x = 2
    epsilon = np.random.normal(scale=var_x**2, size=len(t))
    ys=x_true.copy()
    ys[0,:]+=epsilon
    ys = np.array(ys[0:2,:])
    
    data = pd.DataFrame(index=t)
    data.index.name='time'
    
    data[kf.measurement_columns] = ys.T
        
    yield data


def test_filter(kf, data):

    x0 = np.array([[0,0,g]]).T
    P_0 = np.diag([0.1,0.01,0])
    u = np.array([[]]).T
    result = kf.filter(data=data, P_0=P_0, x0=x0)