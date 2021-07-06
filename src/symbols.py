"""[summary]

[1] Clarke, D., P. Gedling, and G. Hine. “The Application of Manoeuvring Criteria in Hull Design Using Linear Theory.” Transactions of the Royal Institution of Naval Architects, RINA., 1982. https://repository.tudelft.nl/islandora/object/uuid%3A2a5671ac-a502-43e1-9683-f27c50de3570.
[2] Brix, Jochim E. Manoeuvring Technical Manual. Seehafen-Verlag, 1993.

"""

import sympy as sp
from sympy.physics.mechanics import (dynamicsymbols, ReferenceFrame,
                                      Particle, Point)
from src.substitute_dynamic_symbols import lambdify
import pandas as pd
import numpy as np

u, v, r, delta = dynamicsymbols('u v r delta')
m,x_G,U,I_z,volume = sp.symbols('m x_G U I_z volume')
π = sp.pi
T,L,CB,B,rho = sp.symbols('T L CB B rho')

X_lin, Y_lin, N_lin = sp.symbols('X_lin Y_lin N_lin')  #Linearized force models
X_nonlin, Y_nonlin, N_nonlin = sp.symbols('X_nonlin Y_nonlin N_nonlin')  # Nonlinear force models

## Parameters
df_parameters = pd.DataFrame(columns=['symbol','dof','coord','state'])
dofs = ['X','Y','N']
coords = ['u','v','r',r'\delta']
states = ['','dot']

def add_symbol(dof,coord,state=''):

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
    df_parameters.loc[key] = s

for dof in dofs:
    for coord in ['u','v','r']:
        add_symbol(dof=dof,coord=coord, state='dot')


## Add all possible combinations:
from sklearn.preprocessing import PolynomialFeatures
import re 
df_ = pd.DataFrame(columns=['u','v','r','delta'], data=np.zeros((10,4)))
polynomial_features = PolynomialFeatures(degree=3, include_bias=False)
polynomial_features.fit_transform(df_)
feature_names=polynomial_features.get_feature_names(df_.columns)

def rename(result):
    return result.group(1)*int(result.group(2))

feature_names = [re.sub(pattern=r'(\S+)\^(\d)', repl=rename, string=name) for name in feature_names]
feature_names = [name.replace(' ','') for name in feature_names]
for dof in dofs:
    for coord in feature_names:
        add_symbol(dof=dof,coord=coord)

## Parameters according to:
Xudot_ = m / (π*sp.sqrt(L**3/volume)-14) # [Brix] (SI)
import src.prime_system as prime_system
Xudot_prime = Xudot_/prime_system.df_prime.loc['denominator','mass']
df_parameters.loc['Xudot','brix'] =  Xudot_prime # [Brix]
df_parameters.loc['Yvdot','brix'] = -π*(T / L)**2 * (1 + 0.16*CB*B/T - 5.1*(B / L)**2)  # [Clarke]
df_parameters.loc['Yrdot','brix'] = -π*(T / L)**2 * (0.67*B/L - 0.0033*(B/T)**2)  # [Clarke]
df_parameters.loc['Nvdot','brix'] = -π*(T / L)**2 * (1.1*B/L - 0.04*(B/T))  # [Clarke]
df_parameters.loc['Nrdot','brix'] = -π*(T / L)**2 * (1/12 + 0.017*CB*B/T - 0.33*(B/L))  # [Clarke]
df_parameters.loc['Yv','brix'] = -π*(T / L)**2 * (1 + 0.4*CB*B/T)  # [Clarke]
df_parameters.loc['Yr','brix'] = -π*(T / L)**2 * (-1/2 + 2.2*B/L - 0.08*(B/T))  # [Clarke]
df_parameters.loc['Nv','brix'] = -π*(T / L)**2 * (1/2 + 2.4*T/L)  # [Clarke]
df_parameters.loc['Nr','brix'] = -π*(T / L)**2 * (1/4 + 0.039*B/T -0.56*B/L)  # [Clarke]

mask = df_parameters['brix'].notnull()
df_parameters['brix_lambda'] = df_parameters.loc[mask,'brix'].apply(lambdify)


X_qs = sp.Function('X_qs')(u,v,r,delta)  # quasi static force
Y_qs = sp.Function('Y_qs')(u,v,r,delta)  # quasi static force
N_qs = sp.Function('N_qs')(u,v,r,delta)  # quasi static force


