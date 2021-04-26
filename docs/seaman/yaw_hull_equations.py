import sympy as sp
from seaman_symbols import *
import seaman_symbol as ss

yaw_hull_equation = sp.Eq(lhs = N_h.bis,rhs = (N_v.bis + N_r.bis + N_nl.bis))

yaw_drift_equation = sp.Eq(lhs = N_v.bis,rhs = (N_uv + N_uuv*u_w.bis)*u_w.bis*v_w.bis)
yaw_drift_equation_SI = ss.expand_bis(yaw_drift_equation)

yaw_yaw_rate_equation = sp.Eq(lhs = N_r.bis,rhs = (N_ur + N_uur*u_w.bis)*u_w.bis*r_w.bis)
yaw_yaw_rate_equation_SI = ss.expand_bis(yaw_yaw_rate_equation)

T_x = sp.Function('T')(x_s)
f = (-rho / 2 * (v_w + (x_s + Cd_lever*L) * r_w) * sp.Abs(v_w + (x_s + Cd_lever*L) * r_w) * T_x * C_d) * (x_s + Cd_lever*L)
cd_integral = sp.Integral(f, (x_s, -L / 2, L / 2))
yaw_none_linear_equation = sp.Eq(lhs = N_nl,rhs = cd_integral)

k = sp.Symbol('k')
m = sp.Symbol('m')
solution = sp.solve((
            sp.Eq(lhs = m + k*-L/2,rhs = t_a),
            sp.Eq(lhs = m + k*L/2,rhs = t_f),
            ),(m,k), simplify=False)
section_draught_equation = sp.Eq(T_x, solution[k] * (x_s + Cd_lever*L) + solution[m])
simplified_yaw_none_linear_equation = yaw_none_linear_equation.subs(T_x,section_draught_equation.rhs)
simplified_yaw_none_linear_equation_bis = simplified_yaw_none_linear_equation.subs(N_nl,sp.solve(N_nl.bis_eq,N_nl)[0])

yaw_hull_equation_SI = ss.expand_bis(yaw_hull_equation)

yaw_hull_equation_SI = sp.simplify(yaw_hull_equation_SI)
yaw_hull_equation_SI = sp.Eq(N_h,sp.solve(yaw_hull_equation_SI,N_h, simplify=False)[0])

total_yaw_hull_equation_SI = yaw_hull_equation_SI.subs([
    (N_nl,simplified_yaw_none_linear_equation.rhs.as_sum(20)),
    (N_r,sp.solve(yaw_yaw_rate_equation_SI,N_r, simplify=False)[0]),
    (N_v,sp.solve(yaw_drift_equation_SI,N_v, simplify=False)[0])
])