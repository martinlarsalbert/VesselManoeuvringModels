from vessel_manoeuvring_models.prime_system import PrimeSystem
import pytest
import pandas as pd
import numpy as np
import numpy.testing

L = 100
rho = 1025

@pytest.fixture
def ps():
    yield PrimeSystem(L=L,rho=rho)

def test_dict_prime(ps):

    length = 10
    values = {
        'length' : length,
    }
    units = {
        'length' : 'length',
    }
    
    values_prime = ps.prime(values=values, units=units)
    assert values_prime['length'] == length/L

def test_dict_unprime(ps):
        
    length = 10
    values_prime = {
        'length' : length/L,
    }
    units = {
        'length' : 'length',
    }

    values = ps.unprime(values=values_prime, units=units)
    assert values['length'] == length

def test_df_prime(ps):

    length = np.ones(10)*10
    values = pd.DataFrame()
    values['length'] = length
    
    units = {
        'length' : 'length',
    }
    
    values_prime = ps.prime(values=values, units=units)
    numpy.testing.assert_almost_equal(values_prime['length'],length/L)
    