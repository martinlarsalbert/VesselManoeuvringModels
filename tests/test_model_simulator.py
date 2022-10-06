from vessel_manoeuvring_models.models.vmm import ModelSimulator

import vessel_manoeuvring_models.models.linear_vmm as vmm
import pytest
import pandas as pd
import numpy as np
import vessel_manoeuvring_models.prime_system
from vessel_manoeuvring_models.models import brix_coefficients
from vessel_manoeuvring_models.prime_system import PrimeSystem
import os.path

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

@pytest.fixture
def prime_system(ship_parameters):
    yield vessel_manoeuvring_models.prime_system.PrimeSystem(**ship_parameters)

def test_model_simulator(ship_parameters, df_parameters, prime_system):

    ## Define a model simulator:
    # This is a simulator with freezed parameters and ship_parameters:
    parameters = df_parameters['prime']
    model = ModelSimulator(simulator=vmm.simulator, 
                           parameters=parameters, 
                           ship_parameters=ship_parameters, 
                           control_keys=['delta'], 
                           primed_parameters=True, 
                           prime_system=prime_system)


    t = np.linspace(0,10,100)
    df_ = pd.DataFrame(index=t)
    df_['u'] = 10
    df_['v'] = 0
    df_['r'] = 0
    df_['x0' ] = 0
    df_['y0' ] = 0
    df_['psi'] = 0
    df_['delta'] = 0
    df_['U'] = np.sqrt(df_['u']**2 + df_['v']**2)

    result = model.simulate(df_=df_)

def test_model_simulator_save_load(ship_parameters, df_parameters, prime_system,tmpdir):

    ## Define a model simulator:
    # This is a simulator with freezed parameters and ship_parameters:
    parameters = df_parameters['prime']
    model = ModelSimulator(simulator=vmm.simulator, 
                           parameters=parameters, 
                           ship_parameters=ship_parameters, 
                           control_keys=['delta'], 
                           primed_parameters=True, 
                           prime_system=prime_system)

    save_path = os.path.join(str(tmpdir),'model.pkl')
    model.save(path=save_path)

    model2 = ModelSimulator.load(path = save_path)
    assert model.X_eq == model2.X_eq
    assert (model.parameters == model2.parameters).all()

def test_model_simulator_save_simulated(ship_parameters, df_parameters, prime_system,tmpdir):

    ## Define a model simulator:
    # This is a simulator with freezed parameters and ship_parameters:
    parameters = df_parameters['prime']
    model = ModelSimulator(simulator=vmm.simulator, 
                           parameters=parameters, 
                           ship_parameters=ship_parameters, 
                           control_keys=['delta'], 
                           primed_parameters=True, 
                           prime_system=prime_system)

    t = np.linspace(0,10,100)
    df_ = pd.DataFrame(index=t)
    df_['u'] = 10
    df_['v'] = 0
    df_['r'] = 0
    df_['x0' ] = 0
    df_['y0' ] = 0
    df_['psi'] = 0
    df_['delta'] = 0
    df_['U'] = np.sqrt(df_['u']**2 + df_['v']**2)

    result = model.simulate(df_=df_)

    save_path = os.path.join(str(tmpdir),'model.pkl')
    model.save(path=save_path)

