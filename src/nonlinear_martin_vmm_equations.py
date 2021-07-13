"""
References:
[1] : Matusiak, Jerzy. Dynamics of a Rigid Ship - with Applications, 3rd Edition, 2021.
[2] : Triantafyllou, Michael S, and Franz S Hover. “MANEUVERING AND CONTROL OF MARINE VEHICLES.” Massachusetts Institute of Technology, 2003, 152.
"""


import sympy as sp
from src.symbols import *
import pandas as pd
from src.nonlinear_vmm_equations import *

p = df_parameters['symbol']

## X

X_qs_eq = sp.Eq(X_qs,
        p.Xu*u + p.Xuu*u**2 + p.Xuuu*u**3 + p.Xvv*v**2 + 
        #p.Xrr*r**2 + 
        p.Xdeltadelta*delta**2 + 
        p.Xvr*v*r + 
        #p.Xvdelta*v*delta + 
        p.Xrdelta*r*delta + p.Xuvv*u*v**2 + p.Xurr*u*r**2 + p.Xudeltadelta*u*delta**2 + 
        
        p.Xurdelta*u*r*delta + p.Xuvr*u*v*r + 
        #p.Xuvdelta*u*v*delta + 
        p.Xvrdelta*v*r*delta + 
        p.Xthrust*thrust
        )

X_eq = X_eom.subs([
    (X_nonlin,sp.solve(fx_eq,X_nonlin)[0]),
    (X_qs,sp.solve(X_qs_eq,X_qs)[0])
])

## Y

Y_qs_eq = sp.Eq(Y_qs,
        p.Yuu*u**2 + p.Yv*v + 
        #p.Yr*r + 
        p.Ydelta*delta + p.Yudelta*u*delta + p.Yuv*u*v + p.Yur*u*r + p.Yuuv*u**2*v + 
        p.Yuur*u**2*r + p.Yuudelta*u**2*delta + p.Yvvv*v**3 + p.Yrrr*r**3 + 
        #p.Yrrdelta*r**2*delta + 
        p.Yvrr*v*r**2 + 
        p.Yvvr*v**2*r + 
        #p.Yvvdelta*v**2*delta + 
        p.Yvrdelta*v*r*delta + 
        #p.Yrdeltadelta*r*delta**2 + 
        p.Yvdeltadelta*v*delta**2 
    )

Y_eq = Y_eom.subs([
    (Y_nonlin,sp.solve(fy_eq,Y_nonlin)[0]),
    (Y_qs,sp.solve(Y_qs_eq,Y_qs)[0]),
    ])

## N

N_qs_eq = sp.Eq(N_qs,
        p.Nuu*u**2 + p.Nv*v + 
        #p.Nr*r + 
        p.Ndelta*delta + p.Nudelta*u*delta + p.Nuv*u*v + p.Nur*u*r + 
        p.Nuuv*u**2*v + p.Nuur*u**2*r + 
        #p.Nuudelta*u**2*delta + 
        #p.Nvvv*v**3 + 
        p.Nrrr*r**3 + p.Ndeltadeltadelta*delta**3 +
        #p.Nrrdelta*r**2*delta + 
        p.Nvrr*v*r**2 + p.Nvvr*v**2*r + p.Nvvdelta*v**2*delta + p.Nvrdelta*v*r*delta + p.Nrdeltadelta*r*delta**2 + p.Nvdeltadelta*v*delta**2
    )

N_eq = N_eom.subs([
    (N_nonlin,sp.solve(mz_eq,N_nonlin)[0]),
    (N_qs,sp.solve(N_qs_eq,N_qs)[0]),
])
