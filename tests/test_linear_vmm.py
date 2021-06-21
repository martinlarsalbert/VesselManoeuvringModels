import src.models.linear_vmm as model
import pytest
import pandas as pd
from src.symbols import df_prime

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

@pytest.fixture
def df_ship_parameters(ship_parameters):

    df_ship_parameters = pd.DataFrame(data = ship_parameters, index=['value','unit'])
    df_ship_parameters.loc['lambda'] = df_prime.loc['lambda',df_ship_parameters.loc['unit']].values

    yield df_ship_parameters

def test_sim1(df_ship_parameters):

    model.simulate()