import sympy as sp
from sympy import Eq, symbols, Symbol, cos, sin, Derivative, atan, Piecewise, pi, Abs
from vessel_manoeuvring_models.symbols import *
from vessel_manoeuvring_models.parameters import df_parameters
p = df_parameters['symbol']

x_R, y_R, z_R = sp.symbols("x_R y_R z_R")

eq_X_R = Eq(X_R,
            p.X_R0
            + p.X_Rdeltadelta * delta**2
            + p.X_Rvv * v**2
            + p.X_Rrr * r**2
            )

eq_Y_R = Eq(Y_R,
            p.Y_R0 
            + p.Y_Rdelta*delta 
            + p.Y_Rthrustdelta*thrust_propeller*delta
            
            + p.Y_Rv*v
            + p.Y_Rvav*v*Abs(v)
            
            + p.Y_Rr*r
            + p.Y_Rrar*r*Abs(r)
            
            + p.Y_Rvar*v*Abs(r)
            + p.Y_Ravr*Abs(v)*r
            
            + p.Y_Rvrr*v*r**2
            
            + p.Y_Rrdeltadelta*r*delta**2
            + p.Y_Rvdeltadelta*v*delta**2
            + p.Y_Rvrdelta*v*r*delta                        
            
            )
eq_N_R = Eq(N_R,Y_R*x_R)
