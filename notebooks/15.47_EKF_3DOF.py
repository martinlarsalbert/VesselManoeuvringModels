#!/usr/bin/env python
# coding: utf-8

# # Extended Kalman filter for 3 DOF linear model
# An Extended Kalman filter with a 3 DOF linear model as the predictor will be developed.
# The filter is run on simulated data as well as real model test data.

# In[30]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import sympy as sp

import vessel_manoeuvring_models.visualization.book_format as book_format

book_format.set_style()
from vessel_manoeuvring_models.substitute_dynamic_symbols import lambdify
from sympy import Matrix
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Particle, Point
from IPython.display import display, Math, Latex
from vessel_manoeuvring_models.substitute_dynamic_symbols import run, lambdify
from sympy.physics.vector.printing import vpprint, vlatex
from vessel_manoeuvring_models.data import mdl
from vessel_manoeuvring_models.kalman_filter import extended_kalman_filter
import vessel_manoeuvring_models.models.vmm_nonlinear_EOM as vmm
from docs.book.example_1 import ship_parameters, df_parameters
from vessel_manoeuvring_models.symbols import *
from vessel_manoeuvring_models import prime_system

p = df_parameters["symbol"]
from vessel_manoeuvring_models.visualization.plot import track_plot, plot

import matplotlib.pyplot as plt
import os

if os.name == "nt":
    plt.style.use("docs/book/book.mplstyle")  # Windows


# ## 3DOF model

# In[8]:


X_eq = vmm.X_eq
Y_eq = vmm.Y_eq
N_eq = vmm.N_eq

A, b = sp.linear_eq_to_matrix([X_eq, Y_eq, N_eq], [u1d, v1d, r1d])

acceleration = sp.matrices.MutableDenseMatrix([u1d, v1d, r1d])
eq_simulator = sp.Eq(
    sp.UnevaluatedExpr(A) * sp.UnevaluatedExpr(acceleration), sp.UnevaluatedExpr(b)
)
eq_simulator


# In[9]:


A_inv = A.inv()
S = sp.symbols("S")
eq_S = sp.Eq(S, -sp.fraction(A_inv[1, 1])[1])

A_inv_S = A_inv.subs(eq_S.rhs, S)
eq_acceleration_matrix_clean = sp.Eq(
    sp.UnevaluatedExpr(acceleration),
    sp.UnevaluatedExpr(A_inv_S) * sp.UnevaluatedExpr(b),
)
Math(vlatex(eq_acceleration_matrix_clean))


# In[13]:


u1d_function = sp.Function(r"\dot{u}")(u, v, r, delta)
v1d_function = sp.Function(r"\dot{v}")(u, v, r, delta)
r_function = sp.Function(r"\dot{r}")(u, v, r, delta)

subs_prime = [
    (m, m / prime_system.df_prime.mass.denominator),
    (I_z, I_z / prime_system.df_prime.inertia_moment.denominator),
    (x_G, x_G / prime_system.df_prime.length.denominator),
    (u, u / sp.sqrt(u ** 2 + v ** 2)),
    (v, v / sp.sqrt(u ** 2 + v ** 2)),
    (r, r / (sp.sqrt(u ** 2 + v ** 2) / L)),
]

subs = [
    (X_D, vmm.X_qs_eq.rhs),
    (Y_D, vmm.Y_qs_eq.rhs),
    (N_D, vmm.N_qs_eq.rhs),
]

subs = subs + subs_prime

A_SI = A.subs(subs)
b_SI = b.subs(subs)

x_dot = sympy.matrices.dense.matrix_multiply_elementwise(
    A_SI.inv() * b_SI,
    sp.Matrix(
        [(u ** 2 + v ** 2) / L, (u ** 2 + v ** 2) / L, (u ** 2 + v ** 2) / (L ** 2)]
    ),
)


# In[16]:


x_ = sp.Matrix(
    [u * sp.cos(psi) - v * sp.sin(psi), u * sp.sin(psi) + v * sp.cos(psi), r]
)

f_ = sp.Matrix.vstack(x_, x_dot)

subs = {value: key for key, value in p.items()}
subs[psi] = sp.symbols("psi")
lambda_f = lambdify(f_.subs(subs))


# ## Simulation

# In[17]:


def time_step(x_, u_):
    psi = x_[2]
    u = x_[3]
    v = x_[4]
    r = x_[5]
    delta = u_
    x_dot = run(
        lambda_f, **parameters, **ship_parameters, psi=psi, u=u, v=v, r=r, delta=delta
    ).flatten()
    return x_dot


def simulate(x0, E, ws, t, us):

    simdata = np.zeros((6, len(t)))
    x_ = x0

    for i, (u_, w_) in enumerate(zip(us, ws)):

        x_dot = time_step(x_, u_)

        x_ = x_ + h_ * x_dot

        simdata[:, i] = x_

    df = pd.DataFrame(simdata.T, columns=["x0", "y0", "psi", "u", "v", "r"], index=t)
    df.index.name = "time"
    df["delta"] = us

    return df


# In[54]:


parameters = df_parameters["prime"].copy()

N_ = 4000

t_ = np.linspace(0, 50, N_)
h_ = float(t_[1] - t_[0])

us = np.deg2rad(
    30
    * np.concatenate(
        (
            -1 * np.ones(int(N_ / 4)),
            1 * np.ones(int(N_ / 4)),
            -1 * np.ones(int(N_ / 4)),
            1 * np.ones(int(N_ / 4)),
        )
    )
)

np.random.seed(42)
E = np.array([[0, 1]]).T
process_noise = np.deg2rad(0.01)
ws = process_noise * np.random.normal(size=N_)
x0_ = np.array([0, 0, 0, 3, 0, 0])
df = simulate(x0=x0_, E=E, ws=ws, t=t_, us=us)


# In[55]:


track_plot(
    df=df,
    lpp=ship_parameters["L"],
    beam=ship_parameters["B"],
    color="green",
)

plot(df_result=df)


# In[57]:


df_measure = df.copy()

measurement_noise = np.deg2rad(0.5)
epsilon_psi = measurement_noise * np.random.normal(size=N_)

measurement_noise = 0.01
epsilon_x0 = measurement_noise * np.random.normal(size=N_)
epsilon_y0 = measurement_noise * np.random.normal(size=N_)

df_measure["psi"] = df["psi"] + epsilon_psi
df_measure["x0"] = df["x0"] + epsilon_x0
df_measure["y0"] = df["y0"] + epsilon_y0


# ## Kalman filter
# Implementation of the Kalman filter. The code is inspired of this Matlab implementation: [ExEKF.m](https://github.com/cybergalactic/MSS/blob/master/mssExamples/ExEKF.m).

# In[36]:


x, x1d = sp.symbols(r"\vec{x} \dot{\vec{x}}")  # State vector
h = sp.symbols("h")
u_input = sp.symbols(r"u_{input}")  # input vector
w_noise = sp.symbols(r"w_{noise}")  # input vector

f = sp.Function("f")(x, u_input, w_noise)
eq_system = sp.Eq(x1d, f)
eq_system


# In[37]:


eq_x = sp.Eq(x, sp.UnevaluatedExpr(sp.Matrix([x_0, y_0, psi, u, v, r])))
eq_x


# In[44]:


jac = sp.eye(6, 6) + f_.jacobian(eq_x.rhs.doit()) * h
subs = {value: key for key, value in p.items()}
subs[psi] = sp.symbols("psi")
lambda_jacobian = lambdify(jac.subs(subs))


# In[50]:


lambda_jacobian


# In[53]:


def lambda_jacobian_constructor(parameters, ship_parameters):
    def f(x, u):
        delta = u

        psi = x_[2]
        u = x_[3]
        v = x_[4]
        r = x_[5]

        x_dot = run(
            lambda_jacobian,
            **parameters,
            **ship_parameters,
            psi=psi,
            u=u,
            v=v,
            r=r,
            delta=delta
        ).flatten()
        return x_dot

    return f


def lambda_f_constructor(parameters, ship_parameters):
    def f(x, u):
        delta = u

        psi = x_[2]
        u = x_[3]
        v = x_[4]
        r = x_[5]

        x_dot = run(
            lambda_f,
            **parameters,
            **ship_parameters,
            psi=psi,
            u=u,
            v=v,
            r=r,
            delta=delta
        ).flatten()
        return x_dot

    return f


# In[59]:


lambda_jacobian_ = lambda_jacobian_constructor(
    parameters=parameters, ship_parameters=ship_parameters
)

lambda_f_ = lambda_f_constructor(parameters=parameters, ship_parameters=ship_parameters)


# In[70]:


P_prd = np.diag([0.01, 0.01, np.deg2rad(0.1), 0.001, 0.001, np.deg2rad(0.01)])
Qd = np.deg2rad(np.diag([0, 0, 0, 0.5, 0.5, np.deg2rad(0.1)]))
Rd = 0.1

ys = df_measure[["x0", "y0", "psi"]].values

E_ = np.array(
    [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ],
)

C_ = np.array([[1, 1, 1, 0, 0, 0]])

Cd_ = C_
Ed_ = h_ * E_

time_steps = extended_kalman_filter(
    x0=x0_,
    P_prd=P_prd,
    lambda_f=lambda_f_,
    lambda_jacobian=lambda_jacobian_,
    h=h_,
    us=us,
    ys=ys,
    E=E_,
    Qd=Qd,
    Rd=Rd,
    Cd=Cd_,
)
x_hats = np.array([time_step["x_hat"] for time_step in time_steps]).T
time = np.array([time_step["time"] for time_step in time_steps]).T
Ks = np.array([time_step["K"] for time_step in time_steps]).T


# In[ ]:
