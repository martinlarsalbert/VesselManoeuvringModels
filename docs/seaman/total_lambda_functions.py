import sympy as sp
from seaman_symbols import *
import save_lambda_functions as save_lambda_functions
import os
import importlib

directory_path = os.path.dirname(__file__)
from contextlib import contextmanager

@contextmanager
def evaluate_project(project_path):
    """
    Evaluation has the (in)convenience that you always must be standing in the project directory to do stuff.
    This context manager can be used if you want to evaluate a project from another location.
    The context manager visits the project, and then return to where you are.

    Example:
    with evaluate_project('N:\...'):
        sl.load_table_of_content(...)

    :param project_path: Path to where the project is
    :return:
    """

    return_path = os.getcwd()

    try:
        yield os.chdir(project_path)
    except:
        # Do something here...
        os.chdir(return_path)
        raise
    else:
        os.chdir(return_path)

class FunctionProvider():

    def __init__(self,name):

        self.save_name = '%s.py' % name

        self.save_path = os.path.join(directory_path,self.save_name)

        self._function = None
        self.name = name

    def get(self,reevaluate = False,save = True):

        if not reevaluate:
            if os.path.exists(self.save_path):
                return self.load()

        self._function = self.function

        if save:
            self.save()

        return self._function

    @property
    def function(self):
        """
        This function should be replaced by inherited class
        :return:
        """

        return None


    def load(self):

        with evaluate_project(directory_path):
            module = importlib.import_module(name=self.name)
            function = getattr(module, self.name)

        return function


    def save(self):
        save_lambda_functions.save_lambda_to_python_file(lambda_function=self._function,function_name=self.name,
                                                         save_dir=directory_path)


class TotalSurgeFunction(FunctionProvider):

    @property
    def function(self):
        import total_equations as total_equations
        return sp.lambdify((delta, u_w, v_w, r_w, s, T_prop, n_prop, X_res, X_vv, X_rr, X_vr, X_Yrdelta, Y_Tdelta,
                                        Y_uudelta, k_r, k_v, Y_uv, Y_uuv, Y_ur, Y_uur, C_d, t_a, t_f, volume, rho, L,
                                        g,xx_rud,l_cg,n_rud),
                                       sp.solve(total_equations.surge_equation_expanded_SI, X, simplify=False)[0],
                                       modules='numpy',
                                       )


class TotalSwayFunction(FunctionProvider):

    @property
    def function(self):
        import total_equations as total_equations
        return sp.lambdify((delta,u_w,v_w,r_w,s,T_prop, n_prop,Y_Tdelta,Y_uudelta,k_r,k_v,Y_uv,Y_uuv,Y_ur,Y_uur,C_d,t_a,t_f,volume,
                            rho,L,g,xx_rud,l_cg,n_rud),
            sp.solve(total_equations.sway_equation_expanded_SI,Y, simplify=False)[0],
                          modules='numpy',
                          )

class TotalYawFunction(FunctionProvider):

    @property
    def function(self):
        import total_equations as total_equations
        return sp.lambdify((delta,u_w,v_w,r_w,s,T_prop, n_prop,Y_Tdelta,Y_uudelta,k_r,k_v,N_uv,N_uuv,N_ur,N_uur,
                            C_d,t_a,t_f,volume,rho,L,g,xx_rud,l_cg,n_rud,Cd_lever),
            sp.solve(total_equations.yaw_equation_expanded_SI,N, simplify=False)[0],
                          modules='numpy',
                          )

class TotalRollFunction(FunctionProvider):

    @property
    def function(self):
        import total_equations as total_equations
        return sp.lambdify((delta,u_w,v_w,r_w,p,s,T_prop, n_prop,Y_Tdelta,Y_uudelta,k_r,k_v,K_ur,K_uur,K_uv,K_uuv,K_up,K_p,K_vav,
                            K_rar,K_pap,zz_rud,t_a,volume,rho,L,g,xx_rud,l_cg,n_rud),
            sp.solve(total_equations.roll_equation_expanded_SI,K, simplify=False)[0],
                          modules='numpy',
                          )


total_surge_function = TotalSurgeFunction(name = 'X_function').get()
total_sway_function = TotalSwayFunction(name = 'Y_function').get()
total_yaw_function = TotalYawFunction(name = 'N_function').get()
total_roll_function = TotalRollFunction(name = 'K_function').get()
