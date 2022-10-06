"""Module for Linear Vessel Manoeuvring Model (LVMM)
"""
from vessel_manoeuvring_models.symbols import *
from vessel_manoeuvring_models.parameters import *
import vessel_manoeuvring_models.vmm_equations_VCT as eq

from vessel_manoeuvring_models.models.vmm import Simulator
simulator = Simulator(X_eq=eq.X_eq, Y_eq=eq.Y_eq, N_eq=eq.N_eq)
simulator.define_quasi_static_forces(X_qs_eq=eq.X_qs_eq, Y_qs_eq=eq.Y_qs_eq, N_qs_eq=eq.N_qs_eq)