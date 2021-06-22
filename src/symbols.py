import sympy as sp
from sympy.physics.mechanics import (dynamicsymbols, ReferenceFrame,
                                      Particle, Point)
from src.substitute_dynamic_symbols import lambdify
import pandas as pd

u, v, r, delta = dynamicsymbols('u v r delta')
m,x_G,U,I_z = sp.symbols('m x_G U I_z')
π = sp.pi
T,L,CB,B,rho = sp.symbols('T L CB B rho')

X_lin, Y_lin, N_lin = sp.symbols('X_lin Y_lin N_lin')



## Prime System
df_prime = pd.DataFrame()
df_prime.loc['denominator','length'] = L
df_prime.loc['denominator','mass'] = 1/2*rho*L**3
df_prime.loc['denominator','density'] = 1/2*rho
df_prime.loc['denominator','inertia_moment'] = 1/2*rho*L**5
df_prime.loc['denominator','time'] = L/U
df_prime.loc['denominator','area'] = L**2
df_prime.loc['denominator','angle'] = sp.S(1)
df_prime.loc['denominator','-'] = sp.S(1)
df_prime.loc['denominator','linear_velocity'] = U
df_prime.loc['denominator','angular_velocity'] = U/L
df_prime.loc['denominator','linear_acceleration'] = U**2/L
df_prime.loc['denominator','angular_acceleration'] = U**2/L**2
df_prime.loc['denominator','force'] = 1/2*rho*U**2*L**2
df_prime.loc['denominator','moment'] = 1/2*rho*U**2*L**3

df_prime.loc['lambda'] = df_prime.loc['denominator'].apply(lambdify)


## Parameters
df_parameters = pd.DataFrame()
dofs = ['X','Y','N']
coords = ['u','v','r',r'\delta']
states = ['','dot']

for dof in dofs:
    
    for coord in coords:
        for state in states:
            key = f'{dof}{coord}{state}'
            key = key.replace('\\','')
            
            if len(state) > 0:
                symbol_name = r'%s_{\%s{%s}}' % (dof,state,coord)
            else:
                symbol_name = r'%s_{%s%s}' % (dof,state,coord)
            
            s = pd.Series(name=key, dtype='object')
            s['symbol'] = sp.symbols(symbol_name)
            s['dof'] = dof
            s['coord'] = coord
            s['state'] = state
            df_parameters = df_parameters.append(s) 

## Parameters according to (Brix, 1993)
df_parameters.loc['Yvdot','brix'] = -π*(T / L)**2 * (1 + 0.16*CB*B/T - 5.1*(B / L)**2)
df_parameters.loc['Yrdot','brix'] = -π*(T / L)**2 * (0.67*B/L - 0.0033*(B/T)**2)
df_parameters.loc['Nvdot','brix'] = -π*(T / L)**2 * (1.1*B/L - 0.04*(B/T))
df_parameters.loc['Nrdot','brix'] = -π*(T / L)**2 * (1/12 + 0.017*CB*B/T - 0.33*(B/L))
df_parameters.loc['Yv','brix'] = -π*(T / L)**2 * (1 + 0.4*CB*B/T)
df_parameters.loc['Yr','brix'] = -π*(T / L)**2 * (-1/2 + 2.2*B/L - 0.08*(B/T))
df_parameters.loc['Nv','brix'] = -π*(T / L)**2 * (1/2 + 2.4*T/L)
df_parameters.loc['Nr','brix'] = -π*(T / L)**2 * (1/4 + 0.039*B/T -0.56*B/L)

mask = df_parameters['brix'].notnull()
df_parameters['brix_lambda'] = df_parameters.loc[mask,'brix'].apply(lambdify)




