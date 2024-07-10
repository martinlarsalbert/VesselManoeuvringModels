import sympy as sp
from sympy import Eq, symbols

def eq_move_to_LHS(eq:Eq, symbol:sp.Symbol)->Eq:
    
    assert symbol in eq.rhs.args, f"{symbol} cannot be found in the RHS of {eq}"
    return Eq(eq.lhs - symbol,eq.rhs -symbol)

def eq_move_to_RHS(eq:Eq, symbol:sp.Symbol)->Eq:
    
    assert symbol in eq.lhs.args, f"{symbol} cannot be found in the LHS of {eq}"
    return Eq(eq.lhs - symbol,eq.rhs -symbol)

def eq_remove(eq:Eq, symbol:sp.Symbol)->Eq:
    
    if isinstance(symbol,list):
        for s in symbol:
            eq = eq_remove(eq=eq, symbol=s)
        return eq
    else:
        return eq.subs(symbol,0)