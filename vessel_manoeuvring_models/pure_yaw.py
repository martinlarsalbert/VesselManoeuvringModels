import numpy as np


def identify_fourier_coefficients(t:np.ndarray, y:np.ndarray, w0:float, n_terms:int=3, n_periods=1):
    """Identify Fourier coefficients from a measured signal
    Note: The signal should have a complete period exactly, or periods (if n_periods > 1)

    Args:
        t (np.ndarray): time [s]
        y (np.ndarray): signal
        w0 (float): frequency [rad/s]
        n_terms (int, optional): Number of coefficient pairs in series. Defaults to 3.
        n_periods: Number of periods in the signal

    Returns:
        _type_: a0, a_n, b_n, where:
        a0 : (float) intercept
        a_n: a_n[n-1] * np.cos(n * w0 * t)
        b_n: b_n[n-1] * np.sin(n * w0 * t)
        
    """
    
    #w0 = 2*np.pi/T
    #T = 2*np.pi/w0
    
    # Calculate a0
    a0 = (w0/(2*np.pi)) * np.trapz(y, t) / n_periods
    
    # Calculate an and bn for n terms
    a_n = []
    b_n = []
    for n in range(1, n_terms + 1):
        cos_term = np.cos(n * t * w0)
        sin_term = np.sin(n * t * w0)
        an = (w0/np.pi) * np.trapz(y * cos_term, t) / n_periods
        bn = (w0/np.pi) * np.trapz(y * sin_term, t) / n_periods
        a_n.append(an)
        b_n.append(bn)

    return a0, a_n, b_n

def reconstruct_fourier_series(a0:float,a_n:np.ndarray,b_n:np.ndarray,t:np.ndarray,w0:float)->np.ndarray:
    """Reconstruct a Fourier series from coefficients

    Args:
        a0 (float): intercept
        a_n (np.ndarray): a_n[n-1] * np.cos(n * w0 * t)
        b_n (np.ndarray): b_n[n-1] * np.sin(n * w0 * t)_
        t (np.ndarray): time [s]
        w0 (float): frequency [rad/s]

    Returns:
        np.ndarray: reconstructed signal
    """

    N = len(t)
    reconstructed_signal = a0*np.ones(N)
    for n in range(1, len(a_n)+1):
        reconstructed_signal += a_n[n-1] * np.cos(n * w0 * t) + b_n[n-1] * np.sin(n * w0 *t)
    
    return reconstructed_signal

def c_n(a_n, w0, t, n):
    return a_n[n-1] * np.cos(n * w0 * t)

def s_n(b_n, w0, t, n):
    return b_n[n-1] * np.sin(n * w0 * t)