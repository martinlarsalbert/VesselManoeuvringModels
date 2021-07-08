from src.parameters import get_parameter_denominator
from src.symbols import *

def test_get_parameter_denominator_added_mass():

    parameter_denominator = get_parameter_denominator(dof='X', coord='u', state='dot')
    assert parameter_denominator == 1/2*rho*L**3


def test_get_parameter_denominator2():

    parameter_denominator = get_parameter_denominator(dof='X', coord='uvrdelta')
    
    assert parameter_denominator == 1/2*rho*U**2*L**2 / ((U)**2 * U/L)  
