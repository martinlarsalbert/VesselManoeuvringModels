import pytest
from vessel_manoeuvring_models.sigma_points import sigma_points_UKF, sigma_points_CKF
import numpy as np
from numpy.testing import assert_almost_equal


@pytest.fixture
def n():
    n = np.random.randint(1,5)    
    yield n

@pytest.fixture
def x(n):
    x = np.random.random(size=(n,1))
    yield x
    
@pytest.fixture
def P(n):
    P = np.random.random(size=(n,n))
    yield P
    
def test_sigma_points_UKF_random(x,P):
    
    SP,W = sigma_points_UKF(x=x, P=P)
    
    
def test_sigma_points_CKF_random(x,P):
    
    SP,W = sigma_points_CKF(x=x, P=P)
    

def test_sigma_points_UKF():
    
    x = np.array([
    0.957166948242946,
    0.485375648722841,   
    ])
    
    P = np.array([
        [0.818331408407438,	0.499770655255690],
        [0.499770655255690,	0.858703285182333],
    ])
    
    SP_true = np.array([
        [0.957166948242946,	2.44280728441146,	1.45502939588428,	-0.528473387925569,	0.459304500601615],
        [0.485375648722841,	0.983238096364171,	2.01123371511551,	-0.0124867989184890,	-1.04048241766983],
    ])
    
    W_true = np.array([0.333333333333333,	0.166666666666667,	0.166666666666667,	0.166666666666667,	0.166666666666667])
    
    SP,W = sigma_points_UKF(x=x, P=P)

    assert_almost_equal(actual=SP, desired=SP_true)
    assert_almost_equal(actual=W, desired=W_true)
    
def test_sigma_points_CKF():
    
    x = np.array([
    0.792207329559554,
    0.959492426392903,  
    ])
    
    P = np.array([
        [1.15101644261556,	0.816498639230788],
        [0.816498639230788,	0.873618710843284],
    ])
    
    SP_true = np.array([
        [2.16476481951544,	1.43882841025298,	-0.580350160396328,	0.145586248866132],
        [1.60611350708633,	2.11236648702981,	0.312871345699481,	-0.193381634244007],
    ])
    
    W_true = np.array([0.250000000000000,	0.250000000000000,	0.250000000000000,	0.250000000000000])
    
    SP,W = sigma_points_CKF(x=x, P=P)

    assert_almost_equal(actual=SP, desired=SP_true)
    assert_almost_equal(actual=W, desired=W_true)