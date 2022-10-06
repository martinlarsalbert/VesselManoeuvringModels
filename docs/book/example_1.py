
from vessel_manoeuvring_models.parameters import df_parameters
from vessel_manoeuvring_models.substitute_dynamic_symbols import run
from vessel_manoeuvring_models import prime_system

ship_parameters ={'T': 0.2063106796116504,
 'L': 5.014563106796117,
 'CB': 0.45034232324249973,
 'B': 0.9466019417475728,
 'rho': 1000,
 'x_G': 0,
 'm': 441.0267843660858,
 'I_z': 693.124396594905,
 'volume': 0.4410267843660858}

def calculate_prime(row, ship_parameters):
    return run(function=row['brix_lambda'], **ship_parameters)

mask = df_parameters['brix_lambda'].notnull()
df_parameters.loc[mask,'brix_prime'] = df_parameters.loc[mask].apply(calculate_prime, ship_parameters=ship_parameters, axis=1)

df_parameters['prime'] = df_parameters['brix_prime']

df_parameters.loc['Ydelta','prime'] = 0.001  # Just guessing
df_parameters.loc['Ndelta','prime'] = -df_parameters.loc['Ydelta','prime']/2  # Just guessing

df_parameters.loc['Nu','prime'] = 0
df_parameters.loc['Nur','prime'] = 0
df_parameters.loc['Xdelta','prime'] = -0.001
df_parameters.loc['Xr','prime'] = 0
df_parameters.loc['Xrr','prime'] = 0.007
df_parameters.loc['Xu','prime'] = -0.001
df_parameters.loc['Xv','prime'] = 0
df_parameters.loc['Xvr','prime'] = -0.006
df_parameters.loc['Yu','prime'] = 0
df_parameters.loc['Yur','prime'] = 0.001

ps = prime_system.PrimeSystem(**ship_parameters)  # model
ship_parameters_prime = ps.prime(ship_parameters)
