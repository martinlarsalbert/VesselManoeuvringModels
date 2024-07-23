import numpy as np
from scipy.linalg import sqrtm, cholesky

from numpy import zeros, sqrt

def calculate_sigma_points(x:np.ndarray, P:np.ndarray, kind:str='UKF'):
    """Computes sigma points, either using unscented transform or using cubature.

    Args:
        x (np.ndarray): [n x 1] Prior mean
        P (np.ndarray): [n x n] Prior covariance
        kind (str, optional): 'UKF', or 'CKF'. Defaults to 'UKF'.

    Returns:
        SP          [n x 2n+1] UKF, [n x 2n] CKF. Matrix with sigma points
        W           [1 x 2n+1] UKF, [1 x 2n] UKF. Vector with sigma point weights 
    """
   
    if kind=='UKF':
        return sigma_points_UKF(x=x, P=P)

    elif kind=='CKF':
        return sigma_points_CKF(x=x, P=P)
    else:
        ValueError('Incorrect kind of sigma point')

def sigma_points_UKF(x:np.ndarray, P:np.ndarray,):
    
    x = x.flatten()
    n = P.shape[0]
    #P_sqrt = sqrtm(P)
    P_sqrt = cholesky(P)
    
    W_0 = 1 - n/3 # x is Gaussian
    N = 2*n+1

    # Calculate sigma points:
    W = zeros(N)
    SP = zeros((n,N))

    S = sqrt(n/(1-W_0))
    W_n = (1-W_0)/(2*n)
    W[0] = W_0
    SP[:,0] = x.flatten()
    
    for i in range(0,n):
        j = i + 1
        W[j] = W_n
        W[j+n] = W_n

        sigma_point = x + S*P_sqrt[:,i]
        SP[:,j] = sigma_point.flatten()
        sigma_point = x -S*P_sqrt[:,i]
        SP[:,j+n] = sigma_point.flatten()

    return SP,W

def sigma_points_CKF(x:np.ndarray, P:np.ndarray,):
    
    x = x.flatten()
    
    n = P.shape[0]
    P_sqrt = sqrtm(P)
    P_sqrt = cholesky(P)
    
    N = 2*n
            
    # Calculate sigma points:
    W = zeros(N)
    SP = zeros((n,N))
    
    W_n = 1/(2*n)
    S = sqrt(n)
    for i in range(0,n):
        W[i] = W_n
        W[i+n] = W_n
        
        sigma_point = x + S*P_sqrt[:,i]
        SP[:,i] = sigma_point.flatten()
        sigma_point = x -S*P_sqrt[:,i]
        SP[:,i+n] = sigma_point.flatten()
        
    return SP,W