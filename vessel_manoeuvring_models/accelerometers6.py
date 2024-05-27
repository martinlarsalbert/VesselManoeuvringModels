import numpy as np
from numpy.linalg import det, inv

import sympy.physics.mechanics as me
import sympy as sp
from sympy import Eq, symbols
from sympy.physics.vector import ReferenceFrame, Point
from vessel_manoeuvring_models.substitute_dynamic_symbols import run, lambdify
from vessel_manoeuvring_models.symbols import *
from vessel_manoeuvring_models.substitute_dynamic_symbols import equation_to_python_method, expression_to_python_method, eq_dottify
import vessel_manoeuvring_models.reference_frames as reference_frames

def acc(xacc1,yacc1,yacc2,zacc1,zacc2,zacc3,xco,yco,zco,
    point1 = {
    'x_P': 1.625,
    'y_P': 0.025,
    'z_P': -0.564,
    },
    point2 = {
        'x_P': -1.9,
        'y_P': 0.43,
        'z_P': -0.564,
    },
    point3 = {
        'x_P': -1.9,
        'y_P': -0.43,
        'z_P': -0.564,
    },  
        ):
    """

    SSPA Sweden AB
    Lennart Byström 98-10-09

    Routine for calculation of accelarations in the x-, y- and
    z-direction,xdd, ydd and zdd,at an arbitrary point, based
    on measurements from model tests.

    Coordinate system:
    -----------------
    x-axis towards the bow
    y-axis to the starboard
    z-axis downwards

    Indata:

    the 1:st accelerometer measures acceleration in the x-direction
    at a position with coordinates  x1,y1,z1.  It is called 'X1'

    the 2:nd accelerometer measures acceleration in the y-direction
    at a position with coordinates x2,y2,z2


    the 3:rd accelerometer measures acceleration in the y-direction
    at a position with coordinates x3,y3,z3


    the 4:th accelerometer measures acceleration in the z-direction
    at a position with coordinates x4,y4,z4


    the 5:th accelerometer measures acceleration in the z-direction
    at a position with coordinates x5,y5,z5


    the 6:th accelerometer measures acceleration in the z-direction
    at a position with coordinates x6,y6,z6

    xco = x-coordinate of the new position
    yco = y-coordinate of the new position
    zco = z-coordinate of the new position

    -----------coordinates of accelerometers-------------------

    x-axeln längs BL och origo i AP 
    y-axel positiv åt styrbord
    z rel. BL, neg uppåt
  
    """
    

    
    # Accelerometer no 1 measuring in the x-direction
    y1=point1['y_P']
    z1=point1['z_P']
    
    # Accelerometer no 2 and 3 measuring in the y-direction
    x2=point1['x_P']
    z2=point1['z_P']
    #
    x3=point2['x_P']
    z3=point2['z_P']
    
    # Accelerometer no 4,5 and 6 measuring in the z-direction
    x4=point1['x_P']
    y4=point1['y_P']
    x5=point2['x_P']
    y5=point2['y_P']

    x6=point3['x_P']
    y6=point3['y_P']
    
    #   direction     coord
    a=np.array([
    [1, 0, 0,0 ,  z1, y1],#meas. dir. and coord. of 1. accelerom.
    [0, 1, 0,z2,  0 , x2],#meas. dir. and coord. of 2. accelerom
    [0, 1, 0,z3,  0 , x3],
    [0, 0, 1,y4, -x4, 0 ],
    [0, 0, 1,y5, -x5, 0 ],
    [0, 0, 1,y6, -x6, 0 ],
    ])
    
    ierr=0
    eps=np.finfo(float).eps
    if np.abs(det(a)) < eps: #eps is floating-point relative accuracy
       raise ValueError('Matrisen med koordinater är singulär')

    b=inv(a) #invert matrix with directions and accelerometer coordinates

    #  prepare a matrix for calculation of acclerations in
    #  the x-, y- and z-direction
    aa=np.array([
    [1, 0, 0,   0 ,  zco, -yco],
    [0, 1, 0, -zco,   0 ,  xco],
    [0, 0, 1,  yco, -xco,   0 ],
    ]) #matrix with coordinates of 'new point'
        
    #measured accelerations from 6 sensors (this comes from indata to function acc.m)
    #xacc1=xacc1(:) 
    #yacc1=yacc1(:) 
    #yacc2=yacc2(:)
    #zacc1=zacc1(:)
    #zacc2=zacc2(:) 
    #zacc3=zacc3(:) 
    #accel=[xacc1 yacc1 yacc2 zacc1 zacc2 zacc3] #measured accel of sensors
    accel=np.array([xacc1, yacc1, yacc2, zacc1, zacc2, zacc3]) #measured accel of sensors
    

    #CORE PART of program (calculate acc at 'new'  point:
    accref = b @ accel          #b is inverted matrix from above
    c= aa @ accref               #acc at new point
        
    return c



## With 6 accelerometers

## v1d
subs=[
    (reference_frames.x_P,0),
    (reference_frames.y_P,0),
    (reference_frames.z_P,0),
    (reference_frames.theta_,0),
    (q,0),
    
]
eq_y2d_0 = reference_frames.eq_y2d_P.subs(subs)  # Transverse acceleration at the origin
eq_v1d_simplified = Eq(v1d,sp.solve(eq_y2d_0,v1d)[0])  # Calculate sway acceleration (removing the centrepetal force etc.)
lambda_v1d_from_6_accelerometers = expression_to_python_method(eq_v1d_simplified.rhs, function_name='v1d')

## r1d
x_P,y_P,z_P = me.dynamicsymbols("x_P,y_P,z_P")
eq_y_P2d = Eq(y_P.diff().diff(),reference_frames.acceleration_g[1])

x_Ps=[]
y_Ps=[]
z_Ps=[]
n_=2
for i in range(n_):
    x_P_,y_P_,z_P_ = me.dynamicsymbols(f"x_P{i},y_P{i},z_P{i}")
    x_Ps.append(x_P_)
    y_Ps.append(y_P_)
    z_Ps.append(z_P_)

eq_y_P2ds = []
for i in range(n_):
    eq_y_P2ds.append(eq_y_P2d.subs([
        (reference_frames.x_P,x_Ps[i]),
        (reference_frames.y_P,y_Ps[i]),
        (reference_frames.z_P,z_Ps[i]),
        (y_P,y_Ps[i]),
        
    ]    
    ))
    
eq = Eq(eq_y_P2ds[1].lhs-eq_y_P2ds[0].lhs,eq_y_P2ds[1].rhs-eq_y_P2ds[0].rhs, evaluate=False)

eq_2 = eq.subs([
    (z_Ps[0],z_Ps[1],),
    (y_Ps[0],y_Ps[1],), # y_P0==y_P1, same y coord of accelerometers
    (reference_frames.theta.diff().diff(),0),
    (reference_frames.theta.diff(),0),
    (reference_frames.theta,0),
    (reference_frames.phi.diff().diff(),0),
    (reference_frames.phi.diff(),0),
]

)

eq_3 = eq_dottify(Eq(eq_y_P2ds[1].lhs-eq_y_P2ds[0].lhs, eq_2.subs(reference_frames.subs_removing_dynamic_symbols).rhs))
eq_r1d = Eq(r1d,sp.solve(eq_3,r1d)[0])
lambda_r1d_from_6_accelerometers = expression_to_python_method(eq_r1d.rhs, "r1d")