from src.prime_system import PrimeSystem
import pytest

L = 100
rho = 1025

@pytest.fixture
def ps():
    yield PrimeSystem(L=L,rho=rho)

def test_length_prime(ps):

    length = (10, 'length')
    length_prime = ps.prime(length)
    assert length_prime == length[0]/L

def test_length_unprime(ps):
    
    length = (10, 'length')
    length_prime = (length[0]/L, 'length')

    assert length[0] == ps.unprime(length_prime)
    