import sympy as sp
from vessel_manoeuvring_models.symbols import *
from functools import reduce
from operator import add


def eq_abs(deg=1, symbol_prefix="C_x", const=True) -> sympy.core.add.Add:
    """Create a SymPy expression like:
    C_x0 + C_x1*Abs(awa) + C_x2*awa**2 + ...

    Parameters
    ----------
    deg : int, optional
        _description_, by default 1
    symbol_prefix : str, optional
        _description_, by default "C_x"
    const : bool, optional
        _description_, by default True

    Returns
    -------
    sympy.core.add.Add
        _description_
    """

    if const:
        stop = -1
    else:
        stop = 0

    coefficient_symbols = get_coefficient_symbols(
        deg=deg, symbol_prefix=symbol_prefix, const=const
    )

    awa = sp.symbols("awa", real=True)
    return reduce(
        add,
        [
            sp.symbols(f"{symbol}") * sp.Abs(awa**i)
            for i, symbol in zip(range(deg, stop, -1), coefficient_symbols)
        ],
    )


def eq_sign(deg=1, symbol_prefix="C_x", const=True) -> sympy.core.add.Add:
    """Create a SymPy expression like:
    C_x0*awa + C_x1*awa + C_x2*awa*Abs(awa) + ...

    Parameters
    ----------
    deg : int, optional
        _description_, by default 1
    symbol_prefix : str, optional
        _description_, by default "C_x"
    const : bool, optional
        _description_, by default True

    Returns
    -------
    sympy.core.add.Add
        _description_
    """

    if const:
        stop = -1
    else:
        stop = 0

    awa = sp.symbols("awa", real=True)

    coefficient_symbols = get_coefficient_symbols(
        deg=deg, symbol_prefix=symbol_prefix, const=const
    )

    return reduce(
        add,
        [
            sp.symbols(f"{symbol}") * awa * sp.Abs(awa ** (i - 1))
            if np.mod(i, 2) == 0
            else sp.symbols(f"{symbol}") * awa**i
            for i, symbol in zip(range(deg, stop, -1), coefficient_symbols)
        ],
    )


def get_coefficient_symbols(deg=1, symbol_prefix="C_x", const=True):

    if const:
        stop = -1
    else:
        stop = 0

    return [sp.symbols(f"{symbol_prefix}{i}") for i in range(deg, stop, -1)]
