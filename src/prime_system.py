from src.substitute_dynamic_symbols import lambdify,run
from inspect import signature
from src.symbols import *

## Prime System
df_prime = pd.DataFrame()
df_prime.loc['denominator','length'] = L
df_prime.loc['denominator','mass'] = 1/2*rho*L**3
df_prime.loc['denominator','density'] = 1/2*rho
df_prime.loc['denominator','inertia_moment'] = 1/2*rho*L**5
df_prime.loc['denominator','time'] = L/U
df_prime.loc['denominator','area'] = L**2
df_prime.loc['denominator','angle'] = sp.S(1)
df_prime.loc['denominator','-'] = sp.S(1)
df_prime.loc['denominator','linear_velocity'] = U
df_prime.loc['denominator','angular_velocity'] = U/L
df_prime.loc['denominator','linear_acceleration'] = U**2/L
df_prime.loc['denominator','angular_acceleration'] = U**2/L**2
df_prime.loc['denominator','force'] = 1/2*rho*U**2*L**2
df_prime.loc['denominator','moment'] = 1/2*rho*U**2*L**3
df_prime.loc['lambda'] = df_prime.loc['denominator'].apply(lambdify)


class PrimeSystem():

    def __init__(self,L:float, rho:float, **kwargs):
        if isinstance(L,tuple):
            self.L = self.value(L)
            self.rho = self.value(rho)
        else:
            self.L = L
            self.rho = rho
        
    def value(self, item:tuple)->float:
        """Get value for item

        Args:
            item (tuple): (value:float,unit:str)

        Returns:
            float: value for item
        """
        return item[0]

    def unit(self, item:tuple)->str:
        """Get unit for item

        Args:
            item (tuple): (value:float,unit:str)

        Returns:
            str: unit for item
        """
        return item[1]

    def denominator(self, unit:str, U:float=None)->float:
        """Get prime denominator for item

        Args:
            unit (str): (unit)
            U (float) : optionaly add the velocity when that one is needed

        Returns:
            float: denominator for item
        """

        if not unit in df_prime:
            raise ValueError(f'unit:{unit} does not exist')
        lambda_ = df_prime.loc['lambda',unit]
        
        ## U?
        s = signature(lambda_)
        if 'U' in s.parameters.keys():
            if U is None:
                raise ValueError('Please provide the velocity "U"')
            denominator = run(lambda_,L=self.L,rho=self.rho, U=U)
        else:
            denominator = run(lambda_,L=self.L,rho=self.rho)
        
        return denominator

    def prime(self, items, U:float=None)->float:
        """SI -> prime

        Args:
            item (tuple): (value:float,unit:str)
            U (float) : optionaly add the velocity when that one is needed

        Returns:
            float: primed value of item
        """
        return self._work(items=items, U=U, worker=self._prime)

    def unprime(self, items, U:float=None)->float:
        """prime -> SI

        Args:
            item (tuple): (value:float,unit:str)
            U (float) : optionaly add the velocity when that one is needed

        Returns:
            float: SI value of primed item
        """
        return self._work(items=items, U=U, worker=self._unprime)
    
    def _work(self, items, U:float, worker):
        if isinstance(items,tuple):
            return worker(item=items, U=U)
        elif isinstance(items,list):
            return [worker(item=item, U=U) for item in items]
        elif isinstance(items,dict):
            return {key:worker(item=item, U=U) for key,item in items.items()}
        else:
            raise ValueError(f'unable to prime{items}')
        
    def _prime(self, item:tuple, U:float=None)->float:
        """SI -> prime

        Args:
            item (tuple): (value:float,unit:str)
            U (float) : optionaly add the velocity when that one is needed

        Returns:
            float: primed value of item
        """
        unit = self.unit(item=item)
        denominator = self.denominator(unit=unit, U=U)
        value = self.value(item=item)
        
        return value/denominator 

    def _unprime(self, item:tuple, U:float=None)->float:
        """prime -> SI

        Args:
            item (tuple): (value:float,unit:str)

        Returns:
            float: SI value of item
        """
        unit = self.unit(item=item)
        denominator = self.denominator(unit=unit, U=U)
        value = self.value(item=item)
        
        return value*denominator 

    def df_unprime(self, df:pd.DataFrame, units:dict, U:float)->pd.DataFrame:
        """Unprime a dataframe

        Args:
            df (pd.DataFrame): [description]
            units (dict): [description]

        Returns:
            pd.DataFrame: [description]
        """

        df_prime = pd.DataFrame(index=df.index)
        for key,values in df.items():
            unit = units[key]
            denominators = self.denominator(unit=unit, U=U)
            df_prime[key] = values*denominators

        return df_prime



