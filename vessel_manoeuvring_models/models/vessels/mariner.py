"""
References:
[1] Chislett, M. S., and J. Strom-Tejsen. 
    “Planar Motion Mechanis (PMM) Tests and Full Scale Steering and Manoeuvring Predictions for a Mariner Class Vessel.” 
    Hydro- and Aerodynamics Laboratory, Hydrodynamics Section, 
    Lyngby, 
    Denmark, 
    Report No. Hy-6, 
    1965. https://repository.tudelft.nl/islandora/object/uuid%3A6436e92f-2077-4be3-a647-3316d9f16ede.

"""

import sympy as sp
from vessel_manoeuvring_models.symbols import *
from vessel_manoeuvring_models.parameters import *

df_parameters = df_parameters.copy()

## Ship parameters:
m_prime_ = 798e-5
Iz_prime_ = 39.2e-5
xG_prime_ = -0.023

## Hydrodynamic parameters:
p = {
    'Xudot' :  -42e-5,      
    'Xu'    : -184e-5,      
    'Xuu'   : -110e-5,      
    'Xuuu'  : -215e-5,      
    'Xvv'   : -899e-5,      
    'Xrr'   :   18e-5,      
    'Xdeltadelta'   :  -95e-5,      
    'Xudeltadelta'  : -190e-5,      
    'Xvr'   :  798e-5,      
    'Xvdelta'   :   93e-5,      
    'Xuvdelta'  :   93e-5,      

    'Yvdot' :  -748e-5,
    'Yrdot' :-9.354e-5,
    'Yv'    : -1160e-5,
    'Yr'    :  -499e-5,
    'Yvvv'  : -8078e-5,
    'Yvvr'  : 15356e-5,
    'Yuv'   : -1160e-5,
    'Yur'   :  -499e-5,
    'Ydelta'    :   278e-5,
    'Ydeltadeltadelta'  :   -90e-5,
    'Yudelta'   :   556e-5,
    'Yuudelta'  :   278e-5,
    'Yvdeltadelta'  :    -4e-5,
    'Yvvdelta'  :  1190e-5,
    'Y0'    :    -4e-5,
    'Y0u'   :    -8e-5,
    'Y0uu'  :    -4e-5,

    'Nvdot' : 4.646e-5,
    'Nrdot' : -43.8e-5,
    'Nv'    :  -264e-5,
    'Nr'    :  -166e-5,
    'Nvvv'  :  1636e-5,
    'Nvvr'  : -5483e-5,
    'Nuv'   :  -264e-5,
    'Nur'   :  -166e-5,
    'Ndelta'    :  -139e-5,
    'Ndeltadeltadelta'  :    45e-5,
    'Nudelta'   :  -278e-5,
    'Nuudelta'  :  -139e-5,
    'Nvdeltadelta'  :    13e-5,
    'Nvvdelta'  :  -489e-5,
    'N0'    :     3e-5,
    'N0u'   :     6e-5,
    'N0uu'  :     3e-5,

}

extras = (set(p.keys()) - set(df_parameters.index))
if len(extras) > 0:
    raise ValueError(f'The following are extra:{extras}')

df_parameters['prime'] = pd.Series(p)
df_parameters['prime'].fillna(0, inplace=True)
