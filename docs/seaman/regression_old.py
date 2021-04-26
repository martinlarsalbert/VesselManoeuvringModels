import inspect

import pandas as pd
from evaluation.curve_fit import Interpolator

import seaman.docs
from regression import get_inputs


class SympyModel(Interpolator):


    def curve_fit(self, data, sympy_function, **kwargs):

        signature = inspect.signature(self._equation)
        self.parameters_this = list(signature.parameters.keys())
        self.parameters_this.remove('df')

        df_input = get_inputs(function=sympy_function, df=data, parameters_this=self.parameters_this)

        input_data = df_input.copy()
        input_data[self.y_key] = data[self.y_key]

        super().curve_fit(data=input_data,**kwargs)

        signature = inspect.signature(sympy_function)
        self.fixed_columns = set(signature.parameters.keys()) - set(self.parameters.keys())

    def calculate(self, df,ignore_limits = None):

        assert not self.parameters is None

        if ignore_limits is None:
            ignore_limits = self.ignore_limits

        if not ignore_limits:
            self._check_limits(df = df)  # Make sure the df is withing the interpolation range

        df_input = df[self.fixed_columns]

        return self._equation(df=df_input, **self.parameters)


def check_result(result, df):

    mask = ((df['u_w'] == 0) &
            (df['u_w'] == 0)
            )

    result[mask] = 0

    assert pd.notnull(result).all()

    return result


class SurgeModel(SympyModel):

    def curve_fit(self, data,**kwargs):
        sympy_function = total_lambda_functions.total_surge_function
        super().curve_fit(data=data, sympy_function=sympy_function, **kwargs)

    @staticmethod
    def _equation(df, X_vv,X_rr, X_vr,X_Yrdelta,xu,xuu):
        function = total_lambda_functions.total_surge_function

        u = df['u_w']
        if len(df['u_w'].unique()) > 1:
            X_res = xu*u + xuu*u**2
        else:
            X_res = xu * u

        result = function(**df,
                          X_vv = X_vv,
                          X_rr = X_rr,
                          X_vr = X_vr,
                          X_Yrdelta = X_Yrdelta,
                          X_res = X_res,
                          )

        result = check_result(result, df)

        a = 1
        return result

    def calculate(self, df,ignore_limits = None):
        self.fixed_columns-=set(['X_res'])
        return super().calculate(df=df,ignore_limits=ignore_limits)


class SwayModel(SympyModel):

    def curve_fit(self, data,**kwargs):
        sympy_function = total_lambda_functions.total_sway_function
        super().curve_fit(data=data, sympy_function=sympy_function, **kwargs)

    @staticmethod
    def _equation(df, Y_Tdelta, Y_uudelta, Y_uv, Y_ur, Y_uur, C_d):
        function = total_lambda_functions.total_sway_function
        result = function(**df,
                          Y_Tdelta=Y_Tdelta,
                          Y_uudelta=Y_uudelta,
                          Y_uv=Y_uv,
                          #Y_uuv=Y_uuv,
                          Y_ur=Y_ur,
                          Y_uur=Y_uur,
                          C_d=C_d,
                          #s=s
                          )

        result = check_result(result, df)
        return result


class SwayRudderModelKv(SympyModel):

    def curve_fit(self, data, **kwargs):
        """
        Curve fit ONE rudder!
        :param data:
        :param kwargs:
        :return:
        """
        sympy_function = rudder_lambda_functions.rudder_total_sway_function
        super().curve_fit(data=data, sympy_function=sympy_function, **kwargs)

    @staticmethod
    def _equation(df, Y_Tdelta, Y_uudelta, k_v):
        function = rudder_lambda_functions.rudder_total_sway_function  # This is one rudder only!
        result = function(**df,
                          Y_Tdelta=Y_Tdelta,
                          Y_uudelta=Y_uudelta,
                          #s=s,
                          k_v=k_v)

        result = check_result(result, df)

        a = 1
        return result


class YawModel(SympyModel):

    def curve_fit(self, data,**kwargs):
        sympy_function = total_lambda_functions.total_yaw_function
        super().curve_fit(data=data, sympy_function=sympy_function, **kwargs)


    @staticmethod
    def _equation(df, N_uv, N_ur,N_uur,xx_rud,Cd_lever):
        function = total_lambda_functions.total_yaw_function

        result = function(**df,

                 N_uv = N_uv,
                 #N_uuv = N_uuv,
                 N_ur = N_ur,
                 N_uur = N_uur,
                 xx_rud = xx_rud,
                 Cd_lever = Cd_lever)
        result = check_result(result, df)

        a = 1
        return result


class YawHullModel(SympyModel):

    def curve_fit(self, data,**kwargs):
        sympy_function = total_lambda_functions.total_yaw_function
        super().curve_fit(data=data, sympy_function=sympy_function, **kwargs)


    @staticmethod
    def _equation(df, N_uv, N_uuv, N_ur,N_uur,xx_rud,Cd_lever):
        function = total_lambda_functions.total_yaw_function

        result = function(**df,

                 N_uv = N_uv,
                 N_uuv = N_uuv,
                 N_ur = N_ur,
                 N_uur = N_uur,
                 xx_rud = xx_rud,
                 Cd_lever = Cd_lever)
        result = check_result(result, df)

        a = 1
        return result


class RollModel(SympyModel):


    def curve_fit(self, data,**kwargs):
        sympy_function = total_lambda_functions.total_roll_function
        super().curve_fit(data=data, sympy_function=sympy_function, **kwargs)


    @staticmethod
    def _equation(df, K_ur, K_uv):

        function = total_lambda_functions.total_roll_function

        result = function(**df,

                 K_ur = K_ur,
                 #K_uur = K_uur,
                 K_uv = K_uv,
                 #K_uuv = K_uuv,
                 #K_vav = K_vav,
                 #K_rar = K_rar
                )

        result = check_result(result, df)

        return result