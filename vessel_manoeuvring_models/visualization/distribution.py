import numpy as np
from scipy.linalg import sqrtm

def sigmaEllipse2D(mu, Sigma, level=3, npoints=32)->np.ndarray:
        """
        SIGMAELLIPSE2D generates x,y-points which lie on the ellipse describing
        a sigma level in the Gaussian density defined by mean and covariance.
        
        Args:
            mu (np.ndarray): [2 x 1] Mean of the Gaussian density
            Sigma (np.ndarray): [2 x 2] Covariance matrix of the Gaussian density
            level (int, optional): _description_. Which sigma level curve to plot. Can take any positive value, 
                   but common choices are 1, 2 or 3. Default = 3.
            npoints (int, optional): _description_. Number of points on the ellipse to generate. Default = 32.

        Returns:
            np.ndarray: [2 x npoints] matrix. First row holds x-coordinates, second
                   row holds the y-coordinates. First and last columns should 
                   be the same point, to create a closed curve.
        """
                                
        phis=np.linspace(0,2*np.pi,npoints)
        xy = np.zeros((2,npoints))
        for i in range(len(phis)):
            phi=phis[i]
            xy_ = mu + level*sqrtm(Sigma) @ np.array([[np.cos(phi),np.sin(phi)]]).T
            xy[:,i] = xy_.flatten()
        
        return xy