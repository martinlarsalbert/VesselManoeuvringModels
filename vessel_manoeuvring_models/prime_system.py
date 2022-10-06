from vessel_manoeuvring_models.substitute_dynamic_symbols import lambdify, run
from inspect import signature
from vessel_manoeuvring_models.symbols import *
from pandas.api.types import is_numeric_dtype

## Prime System
df_prime = pd.DataFrame()
df_prime.loc["denominator", "length"] = L
df_prime.loc["denominator", "volume"] = L ** 3
df_prime.loc["denominator", "mass"] = sp.Rational(1, 2) * rho * L ** 3
df_prime.loc["denominator", "density"] = sp.Rational(1, 2) * rho
df_prime.loc["denominator", "inertia_moment"] = sp.Rational(1, 2) * rho * L ** 5
df_prime.loc["denominator", "time"] = L / U
df_prime.loc["denominator", "frequency"] = U / L
df_prime.loc["denominator", "area"] = L ** 2
df_prime.loc["denominator", "angle"] = sp.S(1)
df_prime.loc["denominator", "-"] = sp.S(1)
df_prime.loc["denominator", "linear_velocity"] = U
df_prime.loc["denominator", "angular_velocity"] = U / L
df_prime.loc["denominator", "linear_acceleration"] = U ** 2 / L
df_prime.loc["denominator", "angular_acceleration"] = U ** 2 / L ** 2
df_prime.loc["denominator", "force"] = sp.Rational(1, 2) * rho * U ** 2 * L ** 2
df_prime.loc["denominator", "moment"] = sp.Rational(1, 2) * rho * U ** 2 * L ** 3
df_prime.loc["lambda"] = df_prime.loc["denominator"].apply(lambdify)

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

    def prime(self, values: dict, units={}, U: float = None) -> float:
        """SI -> prime

        Args:
            values (dict,pd.Series or pd.DataFrame) : values to convert to prime
            units (dict) : dictionary with description of physical units as string
            U (float) : optionaly add the velocity when that one is needed

        Returns:
            float: primed value of item
        """
        return self._work(values=values, units=units, U=U, worker=self._prime)

    def unprime(self, values: dict, units={}, U: float = None) -> float:
        """prime -> SI

        Args:
            item (tuple): (value:float,unit:str)
            U (float) : optionaly add the velocity when that one is needed

        Returns:
            float: SI value of primed item
        """
        return self._work(values=values, units=units, U=U, worker=self._unprime)

    def _work(self, values: dict, worker, units={}, U: float = None):

        units_ = standard_units.copy()
        units_.update(units)  # add/overide units

        new_values = values.copy()
        for key, value in new_values.items():

            try:
                value / 2  # is this numeric?
            except:

                new_values[key] = value  # for strings etc...
                continue
            else:
                if not key in units_:
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
