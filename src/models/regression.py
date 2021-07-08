import pandas as pd
import numpy as np
import sympy as sp
from src.substitute_dynamic_symbols import lambdify,run
from src import symbols
import re
from src.parameters import df_parameters

class DiffEqToMatrix():
    """This class reformulates a differential equation into a matrix form regression problem:
    y = X*beta + epsilon

    Example:
    Diff eq:
    phi2d + B*phi1d + C*phi = 0

    y      = X     * beta           + epsilon
    -phi.diff().diff() = [B C] x [phi.diff() phi].T  + epsilon

    """
    
    def __init__(self,ode:sp.Eq, label:sp.Symbol, base_features=[]):
        """[summary]

        Args:
            ode (sp.Eq): ordinary differential equation
            label (sp.Symbol) : label <-> dependent variable in regression (usually acceleration)
            base_features : list with base features, ex: [phi] (derivatives phi.diff() and polynomial combinations such as phi.diff()**3 will be figured out)
        """
        
        self.ode = ode
        assert isinstance(self.ode, sp.Eq)
        
        self.label = label
        assert isinstance(self.label, sp.Expr)

        self.base_features = base_features

        self.setup()

    def __repr__(self):
        return str(self.ode)

    def setup(self):
        self.get_acceleration()
        self.get_coefficients()
        self.get_parts()
        self.get_labels_and_features()

    @property
    def X_lambda(self):
        return lambdify(self.eq_X.rhs)

    @property
    def y_lambda(self):
        p = df_parameters['symbol']
        subs = {value:key for key,value in p.items()}
        
        return lambdify(self.eq_y.rhs.subs(subs))

    @property
    def acceleration_lambda(self):
        
        subs = self.feature_names_subs()
        return lambdify(sp.solve(self.acceleration_equation.subs(subs), self.label)[0])

    def feature_names_subs(self):

        ## Rename:
        columns_raw = list(self.eq_beta.rhs)
        subs = {}

        regexp = re.compile(r'\\dot{([^}])+}')

        def replacer(match):
            return r'%sdot' % match.group(1)
        for symbol in columns_raw:

                ascii_symbol = str(symbol)
                ascii_symbol = regexp.sub(repl=replacer, string = ascii_symbol)                       
                ascii_symbol = ascii_symbol.replace('_','')
                ascii_symbol = ascii_symbol.replace('{','')
                ascii_symbol = ascii_symbol.replace('}','')
                ascii_symbol = ascii_symbol.replace('\\','')
                ascii_symbol = ascii_symbol.replace('-','')  # Little bit dangerous
                subs[symbol] = ascii_symbol
        
        return subs 

    def calculate_features(self, data:pd.DataFrame, simplify_names=True):

        X = run(function=self.X_lambda, inputs=data)
        X = X.reshape(X.shape[1],X.shape[-1]).T
        
        subs = self.feature_names_subs()        
        if simplify_names:
            columns = list(subs.values())
        else:
            columns = list(subs.keys())

        X = pd.DataFrame(data=X, index=data.index, columns=columns)
        
        return X

    def calculate_label(self, y:np.ndarray):
        return self.y_lambda(y)

    def get_acceleration(self):
        """Swap around equation to get acceleration in left hand side
        """
        self.acceleration_equation = sp.Eq(self.label,
                                    sp.solve(self.ode, self.label)[0])

    def get_coefficients(self):

        self.coefficients = []

        # Propose derivatives:
        derivatives = []
        for base_feature in self.base_features:
            feature = base_feature.copy()
            for i in range(4):
                derivatives.append(feature)
                feature = feature.diff()

        subs = [(feature,1) for feature in reversed(derivatives)]
        
        for part in self.acceleration_equation.rhs.args:   
            
            coeff = part.subs(subs)
            self.coefficients.append(coeff)

    def get_parts(self):

        self.parts = self.acceleration_equation.rhs.subs([(c,1) for c in self.coefficients]).args
        
    def get_labels_and_features(self):

        self.xs = [sp.symbols(f'x_{i}') for i in range(1,len(self.parts)+1)]
        self.y_ = sp.symbols('y')
        self.X_ = sp.MatrixSymbol('X', 1, len(self.xs))
        self.beta_ = sp.MatrixSymbol('beta', len(self.xs), 1)

        subs = {part:x for part,x in zip(self.parts,self.xs)}

        self.acceleration_equation_x = sp.Eq(self.y_,
                                          self.acceleration_equation.rhs.subs(subs))

        self.eq_beta = sp.Eq(self.beta_,
                       sp.linear_eq_to_matrix([self.acceleration_equation_x.rhs],self.xs)[0].T)

        self.X_matrix = sp.Matrix(list(subs.keys())).T
        self.eq_X = sp.Eq(self.X_, 
                          self.X_matrix)

        self.eq_y = sp.Eq(self.y_,self.label)

def results_summary_to_dataframe(results):
    '''take the result of an statsmodel results table and transforms it into a dataframe'''
    pvals = results.pvalues
    coeff = results.params
    conf_lower = results.conf_int()[0]
    conf_higher = results.conf_int()[1]

    results_df = pd.DataFrame({"$P_{value}$":pvals,
                               "coeff":coeff,
                               "$conf_{lower}$":conf_lower,
                               "$conf_{higher}$":conf_higher
                                })
    
    #Reordering...
    results_df = results_df[["coeff","$P_{value}$","$conf_{lower}$","$conf_{higher}$"]]
    return results_df
