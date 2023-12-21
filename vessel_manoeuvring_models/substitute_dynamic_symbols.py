import sympy as sp
import sympy.physics.mechanics as me
from inspect import signature
import pandas as pd
from sympy.core.numbers import Float
import numpy as np
import inspect
from numpy import pi,sqrt,cos,sin,tan,arctan,log,select,less_equal,nan,greater,sign,array


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


def find_derivatives(dynamic_symbols: set) -> dict:

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
        raise ValueError("%s must be an instance of sympy.Derivative" % derivative)

    order = derivative.args[1][1]
    symbol = derivative.expr

    name = "%s%id" % (symbol.name, order)

    return name


def lambdify(expression, substitute_functions=False):
    new_expression = substitute_dynamic_symbols(expression)

    if substitute_functions:
        ## Replace a all functions "X_D(u,v,r,delta)" with symbols "X_D" etc.
        subs = get_function_subs(new_expression)
        new_expression = new_expression.subs(subs)

    args = new_expression.free_symbols

    # Rearranging to get the parameters in alphabetical order:
    symbol_dict = {symbol.name: symbol for symbol in args}
    symbols = []
    for symbol_name in sorted(symbol_dict.keys()):
        symbols.append(symbol_dict[symbol_name])

    lambda_function = sp.lambdify(symbols, new_expression, modules="numpy")
    return lambda_function


def run(function, inputs={}, **kwargs):
    """Run sympy lambda method
    This one accepts redundant extra parameters (which the sympy lambda does not)
    Warning! This slows down the execution significantly!!!

    Parameters
    ----------
    function : [type]
        [description]
    inputs : dict, optional
        [description], by default {}

    Returns
    -------
    [type]
        [description]
    """
    s = signature(function)
    kwargs.update(inputs)
    parameters = list(s.parameters.keys())
    args = [kwargs[parameter] for parameter in parameters]
    return function(*args)


def get_function_subs(expression):
    """Get a substitution dict to replace a all functions "X_D(u,v,r,delta)" with symbols "X_D" etc."""
    if isinstance(expression, sp.Function) and hasattr(expression, "name"):
        return {expression: expression.name}
    elif isinstance(expression, sp.Derivative):
        # d/du X_D(...)
        # is simplified as "dduX_D":
        simplified_symbol = f"dd{expression.args[1][0]}{expression.args[0].name}"
        return {expression: simplified_symbol}
    else:
        subs = {}
        for part in expression.args:
            subs.update(get_function_subs(part))

    return subs

def remove_functions(expression):
    return expression.subs(get_function_subs(expression))


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
    number_string = np.format_float_positional(
        float(number), precision=precision, unique=False, fractional=False, trim="k"
    )
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
            new_expression = new_expression.subs(
                part, significant(part, precision=precision)
            )
        elif hasattr(part, "args"):
            new_part = _significant_numbers(part, precision=precision)
            new_expression = new_expression.subs(part, new_part)

    return new_expression

def fix_function_for_pickle(eq):
    
    functions = [part for part in eq.rhs.args if isinstance(part,sp.Function)]
    
    for function in functions: 
        function.__class__.__module__ = "__main__"  # Fix for pickle

def expression_to_python_code(expression, function_name:str, substitute_functions=False):
    lambda_ = lambdify(expression=expression, substitute_functions=substitute_functions)
    lines = inspect.getsourcelines(lambda_)[0]
    s = signature(lambda_)
    parameters = list(s.parameters.keys())
    str_parameters = ",".join(parameters)
    if len(str_parameters)>1:
        str_parameters+=","
        
    str_def = f"def {function_name}({str_parameters}**kwargs):\n"
    str_import = "    from numpy import array \n"
    code = str_def + str_import + "".join(lines[1:])
    return code
      
def equation_to_python_code(eq, substitute_functions=False, name=None):
    expression = eq.rhs
    if name is None:
        function_name = str(eq.lhs)
    else:
        function_name = name
    
    return expression_to_python_code(expression=expression, function_name=function_name, substitute_functions=substitute_functions)

def expression_to_python_method(expression, function_name:str, substitute_functions=False):
    exec(expression_to_python_code(expression=expression, function_name=function_name, substitute_functions=substitute_functions))
    return locals()[function_name]

def equation_to_python_method(eq, substitute_functions=False, name=None):  
    expression = eq.rhs
    if name is None:
        function_name = str(eq.lhs)
    else:
        function_name = name
    
    return expression_to_python_method(expression=expression, function_name=function_name, substitute_functions=substitute_functions)