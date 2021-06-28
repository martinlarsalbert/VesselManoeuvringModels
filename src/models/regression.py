import pandas as pd
import numpy as np
import sympy as sp
from src.substitute_dynamic_symbols import lambdify
from src import symbols

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
        return lambdify(self.eq_y.rhs)

    @property
    def acceleration_lambda(self):
        return lambdify(sp.solve(self.acceleration_equation, symbols.phi_dot_dot)[0])

    def calculate_features(self, data:pd.DataFrame):

        X = self.X_lambda(phi=data['phi'], phi1d=data['phi1d'])
        X = X.reshape(X.shape[1],X.shape[-1]).T
        X = pd.DataFrame(data=X, index=data.index, columns=list(self.eq_beta.rhs))

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
