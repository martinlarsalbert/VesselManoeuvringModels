"""
References:
[1] : Wang, Tongtong, Guoyuan Li, Baiheng Wu, Vilmar Æsøy, and Houxiang Zhang. “Parameter Identification of Ship Manoeuvring Model Under Disturbance Using Support Vector Machine Method.” Ships and Offshore Structures, May 19, 2021.
"""


import sympy as sp
from src.symbols import *
import pandas as pd
from src.linear_vmm_equations import *
from src.models.vmm import Simulator

p = df_parameters['symbol']

# Create a simulator for this model:
simulator = Simulator(X_eq=X_eq, Y_eq=Y_eq, N_eq=N_eq)
simulator.define_quasi_static_forces(X_qs_eq=X_qs_eq, Y_qs_eq=Y_qs_eq, N_qs_eq=N_qs_eq)