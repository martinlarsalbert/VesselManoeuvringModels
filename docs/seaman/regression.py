from seaman.docs.notebooks import generate_input
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import inspect


def run_function(df,function):

    parameters_function = generate_input.list_parameters(function)
    return function(**df[parameters_function])


def get_inputs(function,df,parameters_this):

    # Needed parameters:
    parameters_function = set(generate_input.list_parameters(function))
    parameters_equation = set(parameters_this)

    df_parameters_all = set(df.columns)
    avaliable_parameters = df_parameters_all | parameters_equation

    missing_parameters = parameters_function - avaliable_parameters
    #if len(missing_parameters) > 0 :
    #    raise ValueError('Mssing:%s' % missing_parameters)

    df_parameters = (parameters_function & df_parameters_all) - parameters_equation

    df_input = df[df_parameters]
    return df_input

def sympy_to_shipdict_coefficient_name(sympy_name:str):

    s = sympy_name.lower()
    s2 = s.replace('_','')
    s3 = s2.replace('delta','d')
    return s3

class Model():

    y_key = ''
    boundaries = {}

    @property
    def parameters_this(self):
        signature = inspect.signature(self._equation)
        parameters_this = list(signature.parameters.keys())
        parameters_this.remove('df')
        return parameters_this

    @property
    def bounds(self):

        minimums = []
        maximums = []

        for key in self.parameters_this:
            boundaries = self.boundaries.get(key, (-np.inf, np.inf))
            assert len(boundaries) == 2
            minimums.append(boundaries[0])
            maximums.append(boundaries[1])

        return [tuple(minimums), tuple(maximums)]

    def prepare_data(self, data, coefficients:dict):

        df = data.copy()
        for key,value in coefficients.items():
            df[key]=value

        df_input = get_inputs(function=self.function, df=df, parameters_this=self.parameters_this)
        df_input = df_input.astype(float)
        return df_input

    def fit(self, data, coefficients:dict, **kwargs):
        df_input = self.prepare_data(data=data, coefficients=coefficients)
        p0 = 0 * np.ones(len(self.parameters_this), )
        popt, pcov = curve_fit(f=self._equation, xdata=df_input, ydata=data[self.y_key], p0=p0, bounds = self.bounds,
                               **kwargs)

        parameter_values = list(popt)
        self.parameters = dict(zip(self.parameters_this, parameter_values))

    def run(self,result):

        mask = pd.isnull(result)
        result[mask] = 0
        return result

    def calculate(self, df, coefficients:dict, **kwargs):
        df_input = self.prepare_data(data=df, coefficients=coefficients)
        return self.run(self.function(**df_input, **self.parameters))

    def __repr__(self):

        if self.parameters is None:
            return 'not fitted %s' % self.__class__.__name__
        else:
            return '%s %s' % (self.parameters, self.__class__.__name__)



