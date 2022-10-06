# %load imports.py
## Local packages:

%matplotlib inline
%load_ext autoreload
%autoreload 2
%config Completer.use_jedi = False  ## (To fix autocomplete)

## External packages:
import pandas as pd
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
pd.set_option("display.max_columns", None)
import numpy as np
np.set_printoptions(linewidth=150)

import numpy as np
import os
import matplotlib.pyplot as plt
#if os.name == 'nt':
#    plt.style.use('presentation.mplstyle')  # Windows

import plotly.express as px 
import plotly.graph_objects as go

import seaborn as sns
import sympy as sp
from sympy.physics.mechanics import (dynamicsymbols, ReferenceFrame,
                                      Particle, Point)
from sympy.physics.vector.printing import vpprint, vlatex
from IPython.display import display, Math, Latex
from vessel_manoeuvring_models.substitute_dynamic_symbols import run, lambdify

import pyro

import sklearn
import pykalman
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import statsmodels.api as sm

from scipy.integrate import solve_ivp

## Local packages:
from vessel_manoeuvring_models.data import mdl

from vessel_manoeuvring_models.symbols import *
from vessel_manoeuvring_models.parameters import *
import vessel_manoeuvring_models.symbols as symbols
from vessel_manoeuvring_models import prime_system
from vessel_manoeuvring_models.models import regression
from vessel_manoeuvring_models.visualization.regression import show_pred
from vessel_manoeuvring_models.visualization.plot import track_plot

## Load models:
# (Uncomment these for faster loading):
import vessel_manoeuvring_models.models.vmm_martin  as vmm 
from vessel_manoeuvring_models.models.vmm import ModelSimulator
from example_ship2 import ship_parameters, df_parameters


if os.name == 'nt':
    plt.style.use('../docs/book/book.mplstyle')  # Windows
    
from vessel_manoeuvring_models.visualization.plot import track_plot, plot