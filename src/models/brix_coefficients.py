
import src.symbols as symbols
import src.parameters as parameters
from src.substitute_dynamic_symbols import run, lambdify

def calculate_prime(row, df_ship_parameters):
    return run(function=row['brix_lambda'], inputs=df_ship_parameters.loc['value'])

def calculate(df_ship_parameters):

    df_parameters=parameters.df_parameters.copy()
    mask = df_parameters['brix_lambda'].notnull()
    df_parameters.loc[mask,'prime'] = df_parameters.loc[mask].apply(calculate_prime, df_ship_parameters=df_ship_parameters, axis=1)
    
    return df_parameters