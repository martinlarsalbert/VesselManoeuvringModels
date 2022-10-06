
from vessel_manoeuvring_models.parameters import df_parameters
from vessel_manoeuvring_models.substitute_dynamic_symbols import run
from vessel_manoeuvring_models import prime_system
from vessel_manoeuvring_models.data import mdl

meta_data = mdl.load_meta_data(id=22774)
meta_data['rho']=1000
meta_data['mass'] = meta_data['Volume']*meta_data['rho']

T_ = (meta_data.TA + meta_data.TF)/2
L_ = meta_data.lpp
m_ = meta_data.mass
rho_ = meta_data.rho
B_ = meta_data.beam
CB_ = m_/(T_*B_*L_*rho_)
I_z_ = m_*meta_data.KZZ**2
#I_z_ = 900


ship_parameters = {
        'T' : T_,
        'L' : L_,
        'CB' :CB_,
        'B' : B_,
        'rho' : rho_,
        'x_G' : 0,  # motions are expressed at CG
        'm' : m_,
        'I_z': I_z_, 
        'volume':meta_data.Volume,
    }

def calculate_prime(row, ship_parameters):
    return run(function=row['brix_lambda'], inputs=ship_parameters)

mask = df_parameters['brix_lambda'].notnull()
df_parameters.loc[mask,'brix_prime'] = df_parameters.loc[mask].apply(calculate_prime, ship_parameters=ship_parameters, axis=1)

df_parameters['prime'] = df_parameters['brix_prime']

df_parameters.loc['Ydelta','prime'] = 0.001  # Just guessing
df_parameters.loc['Ndelta','prime'] = -df_parameters.loc['Ydelta','prime']/2  # Just guessing

df_parameters.loc['Nur','prime'] = 0.0001
df_parameters.loc['Nvrr','prime'] = 0.0001
df_parameters.loc['Nvvr','prime'] = 0.0001
df_parameters.loc['Xdeltadelta','prime'] = -0.0001
df_parameters.loc['Xrr','prime'] = 0.0025
df_parameters.loc['Xuu','prime'] = -0.001
df_parameters.loc['Xvr','prime'] = -0.001
df_parameters.loc['Xvv','prime'] = -0.001
df_parameters.loc['Yur','prime'] = 0.001
df_parameters.loc['Yvrr','prime'] = 0.001
df_parameters.loc['Yvvr','prime'] = 0
df_parameters.loc['Xthrust','prime'] = 1

ps = prime_system.PrimeSystem(**ship_parameters)  # model
ship_parameters_prime = ps.prime(ship_parameters)

scale_factor = meta_data.scale_factor
ps_ship = prime_system.PrimeSystem(L=ship_parameters['L']*scale_factor, rho=meta_data['rho'])  # ship