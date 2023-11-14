import sympy as sp


def connections(equations: list):
    tree = {}
    for equation in equations:
        # Look for previous symbols in the rhs the equations:
        found_in = list(equation.rhs.free_symbols & set(tree.keys()))
        for symbol in found_in:
            tree[symbol].append(equation)
        # Add symbol to tree
        symbol = equation.lhs
        tree[symbol] = []
    return tree


def find_equations(symbol: sp.Symbol, equations: list, tree={}):
    tree[symbol] = eq = equations[symbol]
    # print(eq)
    for subsymbol in eq.rhs.free_symbols:
        if subsymbol in equations:
            if subsymbol in tree:
                pass
                # print(f"{subsymbol} already found!")
            else:
                # print(f"looking for:{subsymbol}")
                find_equations(subsymbol, equations=equations, tree=tree)

    return tree


def sort_equations(eq_pipeline: list):
    for n in range(0, len(eq_pipeline)):
        for i in range(n + 1, len(eq_pipeline)):
            # if values[n] > values[i]:
            if eq_pipeline[i].lhs in eq_pipeline[n].free_symbols:
                old = eq_pipeline[i]
                eq_pipeline[i] = eq_pipeline[n]
                eq_pipeline[n] = old
