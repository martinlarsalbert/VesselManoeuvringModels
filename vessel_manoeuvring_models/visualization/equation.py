
from IPython.display import Math
from sympy.physics.vector.printing import vlatex
import sympy as sp
from vessel_manoeuvring_models import equations
import re
import numpy as np

equation_dict = dict()
class Equation(Math):
    """Class that helps to keep track of equation labels in jupyter notebook.
    The class makes the correct conversion when:
    * Converted to LaTeX
    * Converted to JupyterBook
    """
        
    def __init__(self,data:sp.Eq,label:str, url=None, filename=None, metadata=None, max_length=150, subs=True):
        self.label = label
        self.expression = data

        if isinstance(data,str):
            data_text = data
        else:            
            data_text = self.eq_to_string(data=data, max_length=max_length, subs=subs)

        super().__init__(data=data_text, url=url, filename=filename, metadata=metadata)

        global equation_dict
        equation_dict[label] = self.expression  # Add this equation to the global list (used for nomenclature)

    def eq_to_string(self, data, max_length=150, subs=True):

        if subs:
            data = data.subs(equations.nicer_LaTeX)

        data_text = vlatex(data)
        if len(data_text) > max_length:
                        
            pattern = r'([+-]*[^+-]+)'
            parts = re.findall(pattern, data_text)
            lengths = np.array([len(part) for part in parts])
            if np.any(lengths > max_length):
                expanded = vlatex(sp.expand(data))
                parts = re.findall(pattern, expanded)
            
            data_text = ''
            row_length = len(data_text)

            for part in parts:
                if (row_length + len(part)) < max_length:
                    data_text+='%s' % part
                    row_length+=len(part)
                else:
                    data_text+='\\\\ %s' % part
                    row_length = len(part)

            data_text_ = '\\begin{aligned}\n%s\n\\end{aligned}' % data_text
        else:
            data_text_ = data_text

        return data_text_


    
    def _repr_latex_(self):
              
        label='eq:one'
        v2 = r"""
\begin{equation}
%s
\label{%s}
\end{equation}
""" % (self.data,self.label)
        
        return Math(v2)._repr_latex_()