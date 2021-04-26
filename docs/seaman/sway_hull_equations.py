import sympy as sp
from seaman_symbols import *
import seaman_symbol as ss

sway_hull_equation = sp.Eq(lhs = Y_h.bis,rhs = (Y_v.bis + Y_r.bis + Y_nl.bis))

sway_drift_equation = sp.Eq(lhs = Y_v.bis,rhs = (Y_uv + Y_uuv*u_w.bis)*u_w.bis*v_w.bis)
sway_drift_equation_SI = ss.expand_bis(sway_drift_equation)

sway_yaw_rate_equation = sp.Eq(lhs = Y_r.bis,rhs = (Y_ur + Y_uur*u_w.bis)*u_w.bis*r_w.bis)
sway_yaw_rate_equation_SI = ss.expand_bis(sway_yaw_rate_equation)

T_x = sp.Function('T')(x_s)
f = (-rho/2*(v_w + x_s*r_w)*sp.Abs(v_w + x_s*r_w)*T_x*C_d)
cd_integral = sp.Integral(f,(x_s,-L/2,L/2))
sway_none_linear_equation = sp.Eq(lhs = Y_nl,rhs = cd_integral)

k = sp.Symbol('k')
m = sp.Symbol('m')
solution = sp.solve((
            sp.Eq(lhs = m + k*-L/2,rhs = t_a),
            sp.Eq(lhs = m + k*L/2,rhs = t_f),
        ),(m,k), simplify=False)
section_draught_equation = sp.Eq(T_x,solution[k]*x_s + solution[m])
simplified_sway_none_linear_equation = sway_none_linear_equation.subs(T_x,section_draught_equation.rhs)
simplified_sway_none_linear_equation_bis = simplified_sway_none_linear_equation.subs(Y_nl,sp.solve(Y_nl.bis_eq,Y_nl)[0])

sway_hull_equation_SI = ss.expand_bis(sway_hull_equation)
sway_hull_equation_SI = sp.simplify(sway_hull_equation_SI)
sway_hull_equation_SI = sp.Eq(Y_h,sp.solve(sway_hull_equation_SI,Y_h, simplify=False)[0])

total_sway_hull_equation_SI = sway_hull_equation_SI.subs([
    (Y_nl,simplified_sway_none_linear_equation.rhs.as_sum(20)),
    (Y_r,sp.solve(sway_yaw_rate_equation_SI,Y_r, simplify=False)[0]),
    (Y_v,sp.solve(sway_drift_equation_SI,Y_v, simplify=False)[0])
])