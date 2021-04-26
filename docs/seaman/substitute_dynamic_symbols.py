import sympy as sp
import sympy.physics.mechanics as me
from inspect import signature
import pandas as pd
from sympy.core.numbers import Float
import numpy as np

def substitute_dynamic_symbols(expression):
    dynamic_symbols = me.find_dynamicsymbols(expression)
    derivatives = find_derivatives(dynamic_symbols)

    derivative_list = []
    # First susbtitute the Derivatives starting with the highest order (since higher order could be broken up in lower order)
    subs = []
    for order in reversed(sorted(derivatives.keys())):
        for derivative in list(derivatives[order]):
            name = find_name(derivative)
            symbol = sp.Symbol(name)
            subs.append((derivative, symbol))
            derivative_list.append(derivative)

    new_expression_derivatives = expression.subs(subs)


    none_derivatives = dynamic_symbols - set(derivative_list)

    # ...Then substitute the dynamic symbols
    subs = []
    for dynamic_symbol in list(none_derivatives):
        name = find_name(dynamic_symbol=dynamic_symbol)
        symbol = sp.Symbol(name)
        subs.append((dynamic_symbol, symbol))

    new_expression = new_expression_derivatives.subs(subs)

    return new_expression


def find_name(dynamic_symbol):
    if isinstance(dynamic_symbol, sp.Derivative):
        name = find_derivative_name(dynamic_symbol)
    else:
        name = dynamic_symbol.name
    return name

def find_derivatives(dynamic_symbols:set)->dict:

    derivatives = {}

    for dynamic_symbol in list(dynamic_symbols):
        if isinstance(dynamic_symbol, sp.Derivative):
            order = dynamic_symbol.args[1][1]

            if not order in derivatives:
                derivatives[order] = []

            derivatives[order].append(dynamic_symbol)

    return derivatives


def find_derivative_name(derivative):

    if not isinstance(derivative, sp.Derivative):
        raise ValueError('%s must be an instance of sympy.Derivative' % derivative)

    order = derivative.args[1][1]
    symbol = derivative.expr

    name = '%s%id' % (symbol.name, order)

    return name


def lambdify(expression):
    new_expression = substitute_dynamic_symbols(expression)
    args = new_expression.free_symbols

    # Rearranging to get the parameters in alphabetical order:
    symbol_dict = {symbol.name: symbol for symbol in args}
    symbols = []
    for symbol_name in sorted(symbol_dict.keys()):
        symbols.append(symbol_dict[symbol_name])

    lambda_function = sp.lambdify(symbols, new_expression, modules='numpy')
    return lambda_function

def run(function,inputs, **kwargs):

    inputs=inputs.copy()

    if isinstance(inputs,dict):
        inputs = pd.Series(inputs)

    constants = pd.Series(dict(**kwargs))

    if isinstance(inputs, pd.Series):
        inputs_columns = inputs.index
    elif isinstance(inputs, pd.DataFrame):
        inputs_columns = inputs.columns
    else:
        raise ValueError('inputs should be wither pd.Series or pd.DataFrame')

    constant_columns = constants.index
    constant_columns = list(set(constant_columns) - set(inputs_columns))
    for constant_column in constant_columns:
        inputs[constant_column] = constants[constant_column]

    s = signature(function)
    input_names = set(s.parameters.keys())
    missing = list(input_names - set(inputs_columns) - set(constant_columns))

    if len(missing) > 0:
        raise ValueError('Sympy lambda function misses:%s' % (missing))

    return function(**inputs[input_names])


def significant(number, precision=3):
    """
    Get the number with significant figures
    Parameters
    ----------
    number
        Sympy Float
    precision
        number of significant figures

    Returns
    -------
        Sympy Float with significant figures.
    """
    number_string = np.format_float_positional(float(number), precision=precision,
                                               unique=False, fractional=False, trim='k')
    return Float(number_string)


def significant_numbers(expression, precision=3):
    """
    Change to a wanted number of significant figures in the expression
    Parameters
    ----------
    expression
        Sympy expression
    precision
        number of significant figures

    Returns
    -------
        Sympy expression with significant figures.
    """
    new_expression = expression.copy()
    return _significant_numbers(new_expression, precision=precision)


def _significant_numbers(new_expression, precision=3):
    for part in new_expression.args:
        if isinstance(part, Float):
            new_expression = new_expression.subs(part, significant(part, precision=precision))
        elif hasattr(part, 'args'):
            new_part = _significant_numbers(part, precision=precision)
            new_expression = new_expression.subs(part, new_part)

    return new_expression