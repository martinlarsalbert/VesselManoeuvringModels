"""
EOM in air
"""


import sympy as sp
from vessel_manoeuvring_models.symbols import *
import pandas as pd
from vessel_manoeuvring_models.nonlinear_vmm_equations import *
from vessel_manoeuvring_models.models.vmm import Simulator

p = df_parameters['symbol']

## X

X_qs_eq = sp.Eq(X_D,f_ext_x             
        )

subs = [
    (p.Xudot,0),
    (p.Xvdot,0),
    (p.Xrdot,0),
    
    (p.Yudot,0),
    (p.Yvdot,0),
    (p.Yrdot,0),
    
    (p.Nudot,0),
    (p.Nvdot,0),
    (p.Nrdot,0),
    

]
fx_eq = fx_eq.subs(subs)

X_eq = X_eom.subs([
    (X_force,sp.solve(fx_eq,X_force)[0]),
    (X_D,sp.solve(X_qs_eq,X_D)[0])
])

## Y

#[1] eq.2-b:
Y_qs_eq = sp.Eq(Y_D,f_ext_y
    )

fy_eq = fy_eq.subs(subs)


Y_eq = Y_eom.subs([
    (Y_force,sp.solve(fy_eq,Y_force)[0]),
    (Y_D,sp.solve(Y_qs_eq,Y_D)[0]),
    ])

## N
#[1] eq.2-c:
N_qs_eq = sp.Eq(N_D,m_ext_z
        
    )

mz_eq = mz_eq.subs(subs)

N_eq = N_eom.subs([
    (N_force,sp.solve(mz_eq,N_force)[0]),
    (N_D,sp.solve(N_qs_eq,N_D)[0]),
])


# Create a simulator for this model:
simulator = Simulator(X_eq=X_eq, Y_eq=Y_eq, N_eq=N_eq)
simulator.define_quasi_static_forces(X_qs_eq=X_qs_eq, Y_qs_eq=Y_qs_eq, N_qs_eq=N_qs_eq)