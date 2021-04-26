"""
The classes in this module extends the Sympy Symbol class, to also carry description, unit and bis denominator
"""

import sympy as sp
from collections import OrderedDict

# my custom class with description attribute
class Symbol(sp.Symbol):
    def __new__(self, name, description='', unit=''):
        obj = sp.Symbol.__new__(self, name)
        obj.description = description
        obj.unit = unit
        return obj


class Coefficient(Symbol):
    def __new__(self, name, description='coefficient', unit=''):
        obj = super().__new__(self, name, description=description, unit=unit)
        return obj

class BisSymbol(Symbol):
    def __new__(self, name,parent_SI_symbol, description='', unit=''):
        obj = super().__new__(self, name, description=description, unit=unit)
        obj.parent_SI_symbol = parent_SI_symbol
        return obj

class Bis(Symbol):

    def __new__(self, name, denominator, description='', unit=''):
        obj = super().__new__(self, name, description=description, unit=unit)
        obj.denominator = denominator

        bis_name = "%s''%s" % (name[0], name[1:])
        obj.bis = BisSymbol(bis_name,parent_SI_symbol = obj, description=description, unit=unit)
        obj.bis_eq = sp.Eq(lhs=obj.bis, rhs=obj / denominator)

        return obj

def expand_bis(equation:sp.Eq):
    """
    Remove all bis symbols from an expression by substituting with the corresponding bis equation
    :param equation: sympy equation
    :return: new equation WITHOUT bis symbols.
    """

    assert isinstance(equation,sp.Eq)
    symbols = equation.lhs.free_symbols | equation.rhs.free_symbols
    subs = []
    for symbol in symbols:
        if isinstance(symbol,BisSymbol):
            subs.append((symbol,symbol.parent_SI_symbol.bis_eq.rhs))

    expanded_equation = equation.subs(subs)
    return expanded_equation

def reduce_bis(equation:sp.Eq):
    """
    Replace symbols with corresponding bis symbols from an expression by substituting with the corresponding bis equation
    :param equation: sympy equation
    :return: new equation WITH bis symbols.
    """

    assert isinstance(equation,sp.Eq)
    symbols = equation.lhs.free_symbols | equation.rhs.free_symbols
    subs = []
    for symbol in symbols:
        if isinstance(symbol,Bis):
            subs.append((symbol,sp.solve(symbol.bis_eq,symbol)[0]))

    reduced = equation.subs(subs)
    return reduced


def create_html_table(symbols:list):


    html = """
        <tr>
            <th>Variable</th>
            <th>Description</th> 
            <th>SI Unit</th> 
        </tr>
        """

    names = [symbol.name for symbol in symbols if isinstance(symbol, sp.Basic)]
    symbols_dict = {symbol.name:symbol for symbol in symbols if isinstance(symbol, sp.Basic)}

    for name in sorted(names):
        symbol = symbols_dict[name]

        if isinstance(symbol, Symbol):
            html_row = """
            <tr>
                <td>$%s$</td>
                <td>%s</td> 
                <td>%s</td> 
            </tr>
            """ % (sp.latex(symbol), symbol.description, symbol.unit)

            html += html_row

    html_table = """
    <table>
    %s
    </table>
    """ % html
    return html_table
