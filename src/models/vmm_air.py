"""
EOM in air
"""


import sympy as sp
from src.symbols import *
import pandas as pd
from src.nonlinear_vmm_equations import *
from src.models.vmm import Simulator

p = df_parameters['symbol']

## X

X_qs_eq = sp.Eq(X_qs,f_ext_x             
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
    (X_qs,sp.solve(X_qs_eq,X_qs)[0])
])

## Y

#[1] eq.2-b:
Y_qs_eq = sp.Eq(Y_qs,f_ext_y
    )

fy_eq = fy_eq.subs(subs)


Y_eq = Y_eom.subs([
    (Y_force,sp.solve(fy_eq,Y_force)[0]),
    (Y_qs,sp.solve(Y_qs_eq,Y_qs)[0]),
    ])

## N
#[1] eq.2-c:
N_qs_eq = sp.Eq(N_qs,m_ext_z
        
    )

mz_eq = mz_eq.subs(subs)

N_eq = N_eom.subs([
    (N_force,sp.solve(mz_eq,N_force)[0]),
    (N_qs,sp.solve(N_qs_eq,N_qs)[0]),
])


# Create a simulator for this model:
simulator = Simulator(X_eq=X_eq, Y_eq=Y_eq, N_eq=N_eq)
simulator.define_quasi_static_forces(X_qs_eq=X_qs_eq, Y_qs_eq=Y_qs_eq, N_qs_eq=N_qs_eq)