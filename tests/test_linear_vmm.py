
import src.models.linear_vmm as model
import pytest
import pandas as pd
import numpy as np
from src.parameters import df_parameters
from src.prime_system import df_prime
from src.models import brix_coefficients
from src.substitute_dynamic_symbols import run, lambdify
from src.prime_system import PrimeSystem

@pytest.fixture
def ship_parameters():
    T_ =10
    L_ = 200
    CB_ = 0.7
    B_ = 30
    rho_ = 1025
    volume_ = T_*B_*L_*CB_
    m_ = volume_*rho_

    yield {
        'T' : T_,
        'L' : L_,
        'CB' :CB_,
        'B' : B_,
        'rho' : rho_,
        'x_G' : 0,
        'm' : m_,
        'I_z': 0.2*m_*L_**2, 
        'volume':volume_,
    }

@pytest.fixture
def df_ship_parameters(ship_parameters):

    df_ship_parameters = pd.DataFrame(data = ship_parameters, index=['value'])
    ps = PrimeSystem(**ship_parameters)
    df_ship_parameters.loc['prime'] = ps.prime(ship_parameters)

    yield df_ship_parameters

@pytest.fixture
def df_parameters(df_ship_parameters):

    df_parameters = brix_coefficients.calculate(df_ship_parameters=df_ship_parameters)
    df_parameters['prime'].fillna(0, inplace=True)
    df_parameters.loc['Ydelta','prime'] = 0.1  # Just guessing
    df_parameters.loc['Ndelta','prime'] = 0.1  # Just guessing



    yield df_parameters

def test_sim1(ship_parameters, df_parameters):


    t = np.linspace(0,10,1000)

    control = {
        'delta' : 0.0,
    }

    y0 = {
    'u' : 10.0, 
    'v' : 0.0,
    'r' : 0.0,
    'x0' : 0,
    'y0' : 0,
    'psi' : 0,
    }

    solution = model.simulate(y0=y0, t=t, df_parameters=df_parameters, ship_parameters=ship_parameters, control=control)

def test_simulation(ship_parameters, df_parameters):

    t = np.linspace(0,10,1000)

    control = {
        'delta' : 0.0,
    }

    y0 = {
    'u' : 10.0, 
    'v' : 0.0,
    'r' : 0.0,
    'x0' : 0,
    'y0' : 0,
    'psi' : 0,
    }

    linear_simulation = model.LinearSimulation()
    solution = linear_simulation.simulate(y0=y0, t=t, df_parameters=df_parameters, ship_parameters=ship_parameters, control=control)
