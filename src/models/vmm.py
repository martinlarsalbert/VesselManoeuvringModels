"""Module for Vessel Manoeuvring Model (VMM) simulation

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sp
from src.symbols import *

from src.parameters import df_parameters
p = df_parameters['symbol']

from src import prime_system

from src.substitute_dynamic_symbols import run, lambdify
from scipy.spatial.transform import Rotation as R
from src import prime_system
from src.visualization.plot import track_plot

class Simulator():


    def __init__(self,X_eq,Y_eq,N_eq):

        self.X_eq = X_eq
        self.Y_eq = Y_eq
        self.N_eq = N_eq

        self.define_EOM(X_eq=X_eq, Y_eq=Y_eq, N_eq=N_eq)


    def define_EOM(self,X_eq:sp.Eq, Y_eq:sp.Eq, N_eq:sp.Eq):
        """Define equation of motion

        Args:
            X_eq (sp.Eq): [description]
            Y_eq (sp.Eq): [description]
            N_eq (sp.Eq): [description]
        """

    
        u1d,v1d,r1d = sp.symbols('u1d, v1d, r1d')
    
        subs = [
            (u.diff(),u1d),
            (v.diff(),v1d),
            (r.diff(),r1d),

        ]
        eq_X_ = X_eq.subs(subs)
        eq_Y_ = Y_eq.subs(subs)
        eq_N_ = N_eq.subs(subs)

        A,b = sp.linear_eq_to_matrix([eq_X_,eq_Y_,eq_N_],[u1d,v1d,r1d])
        self.A = A
        self.b = b
        self.acceleartion_eq = A.inv()*b
        
        ## Lambdify:
        subs = {value:key for key,value in p.items()}
        subs[X_qs] = sp.symbols('X_qs')
        subs[Y_qs] = sp.symbols('Y_qs')
        subs[N_qs] = sp.symbols('N_qs')

        self.acceleration_lambda = lambdify(self.acceleartion_eq.subs(subs))        

    def define_quasi_static_forces(self, X_qs_eq:sp.Eq, Y_qs_eq:sp.Eq, N_qs_eq:sp.Eq):
        """Define the equations for the quasi static forces
        Ex:
        Y_qs(u,v,r,delta) = Yuu*u**2 + Yv*v + Yr*r + p.Ydelta*delta + ...

        Args:
            X_qs_eq (sp.Eq): [description]
            Y_qs_eq (sp.Eq): [description]
            N_qs_eq (sp.Eq): [description]
        """
        self.X_qs_eq = X_qs_eq
        self.Y_qs_eq = Y_qs_eq
        self.N_qs_eq = N_qs_eq
        
        subs = {value:key for key,value in p.items()}        
        self.X_qs_lambda = lambdify(X_qs_eq.rhs.subs(subs))
        self.Y_qs_lambda = lambdify(Y_qs_eq.rhs.subs(subs))
        self.N_qs_lambda = lambdify(N_qs_eq.rhs.subs(subs))
        
         
    
    def step(self, t, states, parameters, ship_parameters, control):

        u,v,r,x0,y0,psi = states

        if u < 0:
            dstates = [
            0,
            0,
            0,
            0,
            0,
            0,
            ]    
            return dstates

        states_dict = {
            'u':u,
            'v':v,
            'r':r,
            'x0':x0,
            'y0':y0,
            'psi':psi,

            }

        inputs = dict(parameters)
        inputs.update(ship_parameters)
        inputs.update(states_dict)

        if isinstance(control, pd.DataFrame):
            index = np.argmin(np.array(np.abs(control.index - t)))
            control_ = dict(control.iloc[index])
        else:
            control_ = control
        inputs.update(control_)

        inputs['U'] = np.sqrt(u**2 + v**2)  #Instantanious velocity

        inputs['X_qs'] = run(function=self.X_qs_lambda, inputs=inputs)
        inputs['Y_qs'] = run(function=self.Y_qs_lambda, inputs=inputs)
        inputs['N_qs'] = run(function=self.N_qs_lambda, inputs=inputs)
        u1d,v1d,r1d = run(function=self.acceleration_lambda, inputs=inputs)

        rotation = R.from_euler('z', psi, degrees=False)
        w = 0
        velocities = rotation.apply([u,v,w])
        x01d = velocities[0]
        y01d = velocities[1]
        psi1d = r    
        dstates = [
            u1d,
            v1d,
            r1d,
            x01d,
            y01d,
            psi1d,
        ]    
        return dstates

    def simulate(self, df_, parameters, ship_parameters, control_keys=['delta','thrust']):
            
        t = df_.index
        t_span = [t.min(),t.max()]
        t_eval = np.linspace(t.min(),t.max(),len(t))

        control_keys = control_keys
        df_control = df_[control_keys]
        ship_parameters = ship_parameters
        parameters = parameters

        df_0 = df_.iloc[0:5].mean(axis=0)
        y0 = {
            'u' : df_0['u'], 
            'v' : df_0['v'],
            'r' : df_0['r'],
            'x0' : df_0['x0'],
            'y0' : df_0['y0'],
            'psi' : df_0['psi']
            }

        solution = solve_ivp(fun=self.step, t_span=t_span, y0=list(y0.values()), t_eval=t_eval, 
                    args=(parameters, ship_parameters, df_control))
        
        
        result = Result(solution=solution, df_model_test=df_, df_control=df_control, ship_parameters=ship_parameters, parameters=parameters, y0=y0)
        return result
    
class Result():

    def __init__(self, solution, df_model_test, df_control, ship_parameters, parameters, y0):
        self.solution=solution
        self.df_model_test=df_model_test
        self.df_control = df_control
        self.ship_parameters = ship_parameters
        self.parameters = parameters
        self.y0=y0

    @property
    def result(self):

        columns = list(self.y0.keys())
        df_result = pd.DataFrame(data=self.solution.y.T, columns=columns)
        df_result.index=self.solution.t

        df_result = df_result.combine_first(self.df_control)

        try:
            df_result['beta'] = -np.arctan2(df_result['v'],df_result['u'])
        except:
            pass

        return df_result

    def plot_compare(self):

        df_result = self.result
        fig,ax = plt.subplots()
        track_plot(df=self.df_model_test, lpp=self.ship_parameters['L'], beam=self.ship_parameters['B'],ax=ax, label='model test')
        track_plot(df=df_result, lpp=self.ship_parameters['L'], beam=self.ship_parameters['B'],ax=ax, label='simulation', color='green')
        ax.legend()

        for key in df_result:
            fig,ax = plt.subplots()
            self.df_model_test.plot(y=key, label='model test', ax=ax)
            df_result.plot(y=key, label='simulation', ax=ax)
            ax.set_ylabel(key)


    


        


