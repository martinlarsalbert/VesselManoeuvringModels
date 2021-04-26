from scipy.optimize import curve_fit
import inspect
import numpy as np
from scipy.optimize import fmin_slsqp
from scipy.integrate import solve_ivp
from abc import ABC, abstractmethod
import pandas as pd
import evaluation.errors as errors
import matplotlib.pyplot as plt
import json
import copy

class LimitError(Exception): pass
class FeatureError(Exception): pass


class Interpolator():

    x_keys = []

    def __init__(self, y_key,maxfev=4000,ignore_limits = False):

        self.parameters = None
        self.y_key = y_key
        self.maxfev=maxfev
        self.p0 = 0.0  # Standard coefficient value
        self.limits = pd.DataFrame()
        self.ignore_limits = ignore_limits

    def __repr__(self):

        if self.parameters is None:
            return 'not fitted %s' % self.__class__.__name__
        else:
            return '%s %s' % (self.parameters, self.__class__.__name__)

    def calculate(self, df,ignore_limits = None):

        assert not self.parameters is None

        if ignore_limits is None:
            ignore_limits = self.ignore_limits

        if not ignore_limits:
            self._check_limits(df = df)  # Make sure the df is withing the interpolation range

        return self._equation(df=df, **self.parameters)

    def _equation(self, df, k):
        raise ValueError('You must define a _equation method')

    def curve_fit(self, data :pd.DataFrame, **kwargs):

        data = data.copy()
        assert isinstance(data,pd.DataFrame)

        if not self.y_key in data:
            raise ValueError('y_key (%s) does not exist in data' % self.y_key)

        missing = set(self.x_keys) - set(data.columns)
        if len(missing) > 0:
            raise ValueError('The following features (x_keys) are missing:%s' % missing)

        self.y_data = data[self.y_key].copy()
        data.drop(columns=self.y_key,inplace=True)

        if len(self.x_keys) > 0:
            self.x_data = data[self.x_keys]
        else:
            self.x_data = data

        self._calculate_limits()

        parameter_names = list(inspect.getargspec(self._equation))[0][1:]

        p0 = self.p0*np.ones(len(parameter_names),)
        try:

            popt, pcov = curve_fit(f=self._equation, xdata=self.x_data,
                               ydata=self.y_data,maxfev=self.maxfev,p0 = p0,**kwargs)


        except KeyError:
            raise FeatureError('Perhaps this key has not been included in x_keys ?')



        parameter_values = list(popt)
        self.parameters = dict(zip(parameter_names, parameter_values))

    def _calculate_limits(self):
        self.limits['min'] = self.x_data.min()
        self.limits['max'] = self.x_data.max()

    def _check_limits(self,df):

        data = df[self.x_keys]
        mask = self.limits['min'] > data
        if mask.any().any():
            raise LimitError("""
            Limits:%s 
            The following values are below the interpolation range (set ignore_limits = True to ignore this) :\n%s
            """ % (self.limits,data[mask]))

        mask = self.limits['max'] < data
        if mask.any().any():
            raise LimitError("""
                        Limits:%s 
                        The following values are above the interpolation range (set ignore_limits = True to ignore this) :\n%s
                        """ % (self.limits, data[mask]))

    def copy(self):
        return copy.deepcopy(self)

    def to_json(self,include_data = True):

        return json.dumps(self.parameters, default=lambda o: getattr(o, '__dict__', str(o)), indent=3)

    def calculate_rms(self, df):

        prediction = self.calculate(df)
        values = df[self.y_key]

        error = (values - prediction)
        rms = np.mean(np.sqrt(error ** 2))

        return rms

    def calculate_rms_normalized(self, df):

        rms = self.calculate_rms(df)
        y = df[self.y_key]
        rms_normalized = rms / (y.max() - y.min())

        return rms_normalized

class Resistance(Interpolator):

    @staticmethod
    def _equation(df, k0, k1, k2, k3, k4, k5):
        V = df['u']

        y = k0 + k1 * V + k2 * V ** 2 + k3 * V ** 3 + k4 * V ** 4 + k5 * V ** 5
        return y









