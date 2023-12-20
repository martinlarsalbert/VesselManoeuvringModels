from vessel_manoeuvring_models.substitute_dynamic_symbols import lambdify, run
from inspect import signature
from vessel_manoeuvring_models.symbols import *
from pandas.api.types import is_numeric_dtype

## Prime System
df_prime = pd.DataFrame()
df_prime.loc["denominator", "length"] = L
df_prime.loc["denominator", "volume"] = L**3
df_prime.loc["denominator", "mass"] = sp.Rational(1, 2) * rho * L**3
df_prime.loc["denominator", "density"] = sp.Rational(1, 2) * rho
df_prime.loc["denominator", "inertia_moment"] = sp.Rational(1, 2) * rho * L**5
df_prime.loc["denominator", "time"] = L / U
df_prime.loc["denominator", "frequency"] = U / L
df_prime.loc["denominator", "area"] = L**2
df_prime.loc["denominator", "angle"] = sp.S(1)
df_prime.loc["denominator", "-"] = sp.S(1)
df_prime.loc["denominator", "linear_velocity"] = U
df_prime.loc["denominator", "linear_velocity**2"] = U**2
df_prime.loc["denominator", "angular_velocity"] = U / L
df_prime.loc["denominator", "linear_acceleration"] = U**2 / L
df_prime.loc["denominator", "angular_acceleration"] = U**2 / L**2
df_prime.loc["denominator", "force"] = sp.Rational(1, 2) * rho * U**2 * L**2
df_prime.loc["denominator", "moment"] = sp.Rational(1, 2) * rho * U**2 * L**3
df_prime.loc["lambda"] = df_prime.loc["denominator"].apply(lambdify)

df_prime.loc["SI unit", "length"] = "m"
df_prime.loc["SI unit", "volume"] = "m^3"
df_prime.loc["SI unit", "mass"] = "kg"
df_prime.loc["SI unit", "density"] = "kg/m^3"
df_prime.loc["SI unit", "inertia_moment"] = "kg*m^2"
df_prime.loc["SI unit", "time"] = "s"
df_prime.loc["SI unit", "frequency"] = "1/s"
df_prime.loc["SI unit", "area"] = "m^2"
df_prime.loc["SI unit", "angle"] = "rad"
df_prime.loc["SI unit", "-"] = "-"
df_prime.loc["SI unit", "linear_velocity"] = "m/s"
df_prime.loc["SI unit", "angular_velocity"] = "rad/s"
df_prime.loc["SI unit", "linear_acceleration"] = "m/s^2"
df_prime.loc["SI unit", "angular_acceleration"] = "rad/s^2"
df_prime.loc["SI unit", "force"] = "N"
df_prime.loc["SI unit", "moment"] = "Nm"

## Standard units:
# (Can be overridden)
standard_units = {
    "T": "length",
    "L": "length",
    "0": "-",
    "CB": "-",
    "B": "length",
    "rho": "density",
    "x_G": "length",
    "m": "mass",
    "I_z": "inertia_moment",
    "delta": "angle",
    "beta": "angle",
    "t": "time",
    "time": "time",
    "u": "linear_velocity",
    "v": "linear_velocity",
    "w": "linear_velocity",
    "p": "angular_velocity",
    "r": "angular_velocity",
    "U": "linear_velocity",
    "V": "linear_velocity",
    "u1d": "linear_acceleration",
    "v1d": "linear_acceleration",
    "r1d": "angular_acceleration",
    "x0": "length",
    "y0": "length",
    "z0": "length",
    "x01d": "linear_velocity",
    "y01d": "linear_velocity",
    "z01d": "linear_velocity",
    "x02d": "linear_acceleration",
    "y02d": "linear_acceleration",
    "z02d": "linear_acceleration",
    "psi": "angle",
    "phi": "angle",
    "dx0": "linear_velocity",
    "dy0": "linear_velocity",
    "x01d": "linear_velocity",
    "y01d": "linear_velocity",
    "psi1d": "angular_velocity",
    "psi2d": "angular_acceleration",
    "fx": "force",
    "fy": "force",
    "fz": "force",
    "X": "force",
    "Y": "force",
    "Z": "force",
    "mx": "moment",
    "my": "moment",
    "mz": "moment",
    "N": "moment",
    "thrust": "force",
    "torque": "moment",
    "X_qs": "force",
    "Y_qs": "force",
    "N_qs": "moment",
    "volume": "volume",
    "id": "-",
    "scale_factor": "-",
    "x_r": "length",
    "x_p": "length",
    "y_p": "length",
    "z_p": "length",
    "y_p_port": "length",
    "y_p_stbd": "length",
    "Xudot": "mass",
    "Yvdot": "mass",
    "Yrdot": "mass",
    "Nvdot": "inertia_moment",
    "Nrdot": "inertia_moment",
    "U": "linear_velocity",
    "D": "length",
    "tdf": "-",
    "C_2_beta_p_pos": "-",
    "C_2_beta_p_neg": "-",
    "w_p0": "-",
    "C_1": "-",
    "X_R": "force",
    "Y_R": "force",
    "N_R": "moment",
    "A_R": "area",
    "H_R": "length",
    "C_R": "length",
    "rev": "frequency",
    "TWIN": "-",
    "w_p": "-",
    "beta_p": "angle",
    r"Arr/Ind/Fri": "-",
    "awa": "angle",
    "aws": "linear_velocity",
    "twa": "angle",
    "tws": "linear_velocity",
    "A_XV": "area",
    "A_YV": "area",
    "rho_A": "density",
    "cog": "angle",
    "g": "linear_acceleration",
    "n_prop": "-",
    "x_R": "length",
    "y_R": "length",
    "z_R": "length",
    "A_R": "area",
    "b_R": "length",
    "w_f": "-",
    "r_0": "length",
    "x": "length",
    "X_H": "force",
    "Y_H": "force",
    "N_H": "moment",
    "X_P": "force",
    "Y_P": "force",
    "N_P": "moment",
    "X_R": "force",
    "Y_R": "force",
    "N_R": "moment",
    "X_W": "force",
    "Y_W": "force",
    "N_W": "moment",
    "X_D": "force",
    "Y_D": "force",
    "N_D": "moment",
    "V_x": "linear_velocity",
    "C_L": "-",
    "C_D": "-",
    "X_WC": "force",
    "Y_WC": "force",
    "N_WC": "moment",
    "alfa_F": "angle",
    "L_F": "force",
    "D_F": "force",
    "X_RHI": "force",
    "Y_RHI": "force",
    "N_RHI": "moment",
    "V_A": "linear_velocity",
    "C_Th": "-",
    "c_r": "length",
    "c_t": "length",
    "c": "length",
    "n_rudd": "-",
    "X_R_port": "force",
    "Y_R_port": "force",
    "N_R_port": "moment",
    "X_R_stbd": "force",
    "Y_R_stbd": "force",
    "N_R_stbd": "moment",
    "thrust_port": "force",
    "thrust_stbd": "force",
    "X_P_port": "force",
    "X_P_stbd": "force",
    "Y_P_port": "force",
    "Y_P_stbd": "force",
    "N_P_port": "moment",
    "N_P_stbd": "moment",
    "x_fan_aft": "length",
    "x_fan_fore": "length",
    "y_fan_aft": "length",
    "y_fan_fore": "length",
    "F_aftfan": "force",
    "F_forefan": "force",
    "alpha_aftfan": "angle",
    "alpha_forefan": "angle",
    "A_R_C": "area",
    "A_R_U": "area",
    "x_aftfan": "length",
    "x_forefan": "length",
    "y_R_port": "length",
    "y_R_stbd": "length",
    "alpha_port":'angle',
    "alpha_stbd":"angle",
    "V_R_C_port":"linear_velocity",
    "V_R_C_stbd":"linear_velocity",
    "V_R_U_port":"linear_velocity",
    "V_R_U_stbd":"linear_velocity",
    "u_R_port":"linear_velocity",
    "v_R_port":"linear_velocity",
    "w_R_port":"linear_velocity",
    "u_R_stbd":"linear_velocity",
    "v_R_stbd":"linear_velocity",
    "w_R_stbd":"linear_velocity",
    
}


def get_denominator(key: str = None, unit: str = None, output="denominator"):
    """Get prime denominator for item

    Args:
        unit (str): (unit)
        output : specify output

    Returns: denominator as sympy expression if output=='denominator'
           : denominator as python method if output=='lambda'
    """

    if key is None:
        if unit is None:
            raise ValueError("both key and unit cannot be None")

    else:
        if unit is None:
            unit = get_unit(key)

    if not unit in df_prime:
        raise ValueError(f"unit:{unit} does not exist")

    denominator = df_prime.loc[output, unit]
    return denominator


def get_unit(key):
    if not key in standard_units:
        raise ValueError(f"Please define a unit for {key}")

    return standard_units[key]


class PrimeSystem:
    def __init__(self, L: float, rho: float, **kwargs):
        if isinstance(L, tuple):
            self.L = self.value(L)
            self.rho = self.value(rho)
        else:
            self.L = L
            self.rho = rho

    def __repr__(self):
        return f"L={self.L}, rho={self.rho}"

    def denominator(self, unit: str, U: float = None) -> float:
        """Get prime denominator for item

        Args:
            unit (str): (unit)
            U (float) : optionaly add the velocity when that one is needed
            Returns:
            float: denominator for item
        """

        if not unit in df_prime:
            raise ValueError(f"unit:{unit} does not exist")
        lambda_ = df_prime.loc["lambda", unit]

        ## U?
        s = signature(lambda_)
        if "U" in s.parameters.keys():
            if U is None:
                raise ValueError('Please provide the velocity "U"')
            denominator = run(lambda_, L=self.L, rho=self.rho, U=U)
        else:
            denominator = run(lambda_, L=self.L, rho=self.rho)

        return denominator

    def prime(self, values: dict, units={}, U: float = None, only_with_defined_units=False) -> float:
        """SI -> prime

        Args:
            values (dict,pd.Series or pd.DataFrame) : values to convert to prime
            units (dict) : dictionary with description of physical units as string
            U (float) : optionaly add the velocity when that one is needed

        Returns:
            float: primed value of item
        """
        return self._work(values=values, units=units, U=U, worker=self._prime, only_with_defined_units=only_with_defined_units)

    def unprime(self, values: dict, units={}, U: float = None, only_with_defined_units=False) -> float:
        """prime -> SI

        Args:
            item (tuple): (value:float,unit:str)
            U (float) : optionaly add the velocity when that one is needed

        Returns:
            float: SI value of primed item
        """
        return self._work(values=values, units=units, U=U, worker=self._unprime, only_with_defined_units=only_with_defined_units)

    def _work(self, values: dict, worker, units={}, U: float = None, only_with_defined_units=False):
        
                        
        units_ = standard_units.copy()
        units_.update(units)  # add/overide units

        
        if isinstance(values, pd.DataFrame):
            numeric_values = values.select_dtypes(include='number')
            columns_others = list(set(values.columns) - set(numeric_values.columns))
            new_values = values[columns_others].copy()
        else:
            new_values = {}
            
        for key, value in values.items():
            if key in new_values:
                continue
                                    
            if not key in units_:
                if only_with_defined_units:
                    if key in new_values:
                        new_values.drop(columns=key)
                    
                    continue
                else:
                    raise ValueError(f"Please define a unit for {key}")

            unit = units_[key]
            new_values[key] = worker(value=value, unit=unit, U=U)

        return new_values

    def _prime(self, value: float, unit: str, U: float = None) -> float:
        """SI -> prime

        Args:
            value : the value to convert to prime
            unit : physical unit ex:'length' etc.
            U (float) : optionaly add the velocity when that one is needed

        Returns:
            float: primed value of item
        """

        denominator = self.denominator(unit=unit, U=U)
        return value / denominator

    def _unprime(self, value: float, unit: str, U: float = None) -> float:
        """prime -> SI

        Args:
            value : the value to convert from prime
            unit : physical unit ex:'length' etc.

        Returns:
            float: SI value of item
        """
        denominator = self.denominator(unit=unit, U=U)
        return value * denominator

    def df_unprime(self, df: pd.DataFrame, units: dict, U: float) -> pd.DataFrame:
        """Unprime a dataframe

        Args:
            df (pd.DataFrame): [description]
            units (dict): [description]

        Returns:
            pd.DataFrame: [description]
        """

        df_prime = pd.DataFrame(index=df.index)
        for key, values in df.items():
            unit = units[key]
            denominators = self.denominator(unit=unit, U=U)
            df_prime[key] = values * denominators

        return df_prime
