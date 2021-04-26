import sympy as sp
from seaman.docs.notebooks.seaman_symbols import *
import seaman.docs.notebooks.seaman_symbol as ss

surge_hull_equation = sp.Eq(lhs = X_h.bis,rhs = (X_vr*v_w.bis*r_w.bis + X_vv*v_w.bis**2 + X_rr*r_w.bis**2 + X_res.bis))
surge_hull_equation_SI = ss.expand_bis(surge_hull_equation)