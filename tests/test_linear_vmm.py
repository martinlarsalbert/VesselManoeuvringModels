
import src.models.linear_vmm as model
import pytest
import pandas as pd
import numpy as np
from src.symbols import df_prime, df_parameters
from src.models import brix_coefficients
from src.substitute_dynamic_symbols import run, lambdify

@pytest.fixture
def ship_parameters():
    T_ =10
    L_ = 200
    CB_ = 0.7
    B_ = 30
    rho_ = 1025
    m_ = T_*B_*L_*CB_*rho_

    yield {
        'T' : (T_,'length'),
        'L' : (L_,'length'),
        'CB' :(CB_,'-'),
        'B' : (B_,'length'),
        'rho' : (rho_,'density'),
        'x_G' : (0,'length'),
        'm' : (m_,'mass'),
        'I_z': (0.2*m_*L_**2, 'inertia_moment'),

    }


def calculate_prime_ship(col, df_ship_parameters):
    denominator = run(function=col['lambda'], inputs=df_ship_parameters.loc['value'])
    prime = col['value'] / denominator
    return prime


@pytest.fixture
def df_ship_parameters(ship_parameters):

    df_ship_parameters = pd.DataFrame(data = ship_parameters, index=['value','unit'])
    df_ship_parameters.loc['lambda'] = df_prime.loc['lambda',df_ship_parameters.loc['unit']].values

    for key in ['denominator','lambda']:
        df_ship_parameters.loc[key] = df_prime[df_ship_parameters.loc['unit']].loc[key].values
    
    df_ship_parameters.loc['prime'] = df_ship_parameters.apply(func=calculate_prime_ship, axis=0, df_ship_parameters=df_ship_parameters)

    yield df_ship_parameters

@pytest.fixture
def df_parameters(df_ship_parameters):

    df_parameters = brix_coefficients.calculate(df_ship_parameters=df_ship_parameters)
    df_parameters['prime'].fillna(0, inplace=True)
    df_parameters.loc['Ydelta','prime'] = 0.1  # Just guessing
    df_parameters.loc['Ndelta','prime'] = 0.1  # Just guessing



    yield df_parameters

def test_sim1(df_ship_parameters, df_parameters):


    t = np.linspace(0,10,1000)


    control = {
        'delta' : (0.0, 'angle'),
    }

    y0 = [
        10.0,  ## u
        0.0,  ## v
        0.0,  ## r
        #0,  ## ud1
        #0,  ## vd1
        #0,  ## rd1
    ]

    


    solution = model.simulate(y0=y0, t=t, df_parameters=df_parameters, df_ship_parameters=df_ship_parameters, control=control)