import numpy as np

## 2d line interpolation

def point_between(x_start: float, x_end:float, y_start:float, y_end:float, dS:float):
    """Find a point between start and end of a line at the distance dS of this line. 

    Args:
        x_start (float): _description_
        x_end (float): _description_
        y_start (float): _description_
        y_end (float): _description_
        dS (float): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: the point vector [x,y]
    """

    dx = x_end - x_start
    dy = y_end - y_start
    S = np.sqrt(dx**2 + dy**2)

    if dS > S*1.00001:
        raise ValueError(f"dS ({dS}) is larger than the line length ({S})")

    point = np.array([
        x_start + dx/S*dS,
        y_start + dy/S*dS,
    ])

    return point

def interp_line_(xs:np.ndarray,ys:np.ndarray, S:float=None, fraction:float=None):
    """Find the interpolated point along line segments xs, ys

    Args:
        xs (np.ndarray): _description_
        ys (np.ndarray): _description_
        S (float, optional): _description_. Defaults to None.
        fraction (float, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: the point vector [x,y]
    """

    xs = np.array(xs)
    ys = np.array(ys)
    
    dSs = np.sqrt(np.diff(xs)**2+np.diff(ys)**2)
    Ss = np.concatenate(([0.0],np.cumsum(dSs)))
    
    if S is None:
        if fraction is None:
            raise ValueError('Specify either "S" or "fraction"')

        assert ((fraction >= 0) & (fraction<=1))
        S = Ss[-1]*fraction

    mask = (S >= Ss[0:-1]) & (S < Ss[1:])    

    founds = np.sum(mask)
    if founds==1:
        i = np.argmax(mask)
    elif founds==0:
        i = len(Ss)-2 ## Special case
    else:
        raise ValueError(f"Too many founds: {founds}")
    
    dS = S - Ss[i] #Ss[i+1] - S

    point = point_between(x_start=xs[i], x_end=xs[i+1], y_start=ys[i], y_end=ys[i+1], dS=dS)

    return point

def interp_line(xs:np.ndarray,ys:np.ndarray, S:float=None, fraction:float=None):
    """Find the interpolated point(s) along line segments xs, ys

    Args:
        xs (np.ndarray): _description_
        ys (np.ndarray): _description_
        S:
            None -> Use "fraction" instead
            float -> interpolate for one scalar S
            array -> interpolate for several S
            
        fraction:
            None -> Use "S" instead
            float -> interpolate for one scalar fraction
            array -> interpolate for several fractions
    Returns:
        _type_: either [x,y] for one point or a vector [x,y].T
    """

    if isinstance(S,float) or isinstance(fraction,float):
        return interp_line_(xs=xs, ys=ys, S=S, fraction=fraction)

    if S is None:
        points = [interp_line_(xs=xs, ys=ys, S=S, fraction=f) for f in fraction]
    else:
        points = [interp_line_(xs=xs, ys=ys, S=s, fraction=fraction) for s in S]

    return np.array(points)