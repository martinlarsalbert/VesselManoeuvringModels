import sympy.physics.mechanics as me
import sympy as sp
from sympy import Eq, symbols
from sympy.physics.vector import ReferenceFrame, Point
from vessel_manoeuvring_models.substitute_dynamic_symbols import run, lambdify
from vessel_manoeuvring_models.symbols import *
from vessel_manoeuvring_models.substitute_dynamic_symbols import equation_to_python_method, expression_to_python_method

theta, phi, psi, V = me.dynamicsymbols("theta,phi,psi,V")
x_0, y_0 = me.dynamicsymbols("x_0 y_0")

I = ReferenceFrame("I")  # Inertial frame
B = ReferenceFrame("B")  # Body frame
B.orient(parent=I, rot_type="Body", amounts=(psi, theta, phi), rot_order="321")

O = Point("O")  # World Origo, fixed in I
O.set_vel(I, 0)

O_B = Point("O_B")  # Origo of B, Point fixed in B with known velocity in I.
O_B.set_pos(O, x_0 * I.x + y_0 * I.y)
u, v = me.dynamicsymbols("u v")
O_B.set_vel(I, u * B.x + v * B.y)


## Move GPS position to origo:
x_GPS_I, y_GPS_I, z_GPS_I = sp.symbols("x_GPS_I,y_GPS_I,z_GPS_I")
P_GPS = Point("P_GPS")  # Point of the GPS, fixed in B
x_GPS_B, y_GPS_B, z_GPS_B = sp.symbols("x_GPS_B,y_GPS_B,z_GPS_B")
P_GPS.set_pos(O_B, x_GPS_B * B.x + y_GPS_B * B.y + z_GPS_B * B.z)

eq_P_GPS = sp.Eq(
    (x_GPS_I * I.x + y_GPS_I * I.y + z_GPS_I * I.z).to_matrix(I),
    P_GPS.pos_from(O).express(I).to_matrix(I),
)

# Origo position from GPS measurement:
eq_x_0 = sp.Eq(x_0, sp.solve(sp.Eq(eq_P_GPS.lhs[0], eq_P_GPS.rhs[0]), x_0)[0])
eq_y_0 = sp.Eq(y_0, sp.solve(sp.Eq(eq_P_GPS.lhs[1], eq_P_GPS.rhs[1]), y_0)[0])

lambda_x_0 = lambdify(eq_x_0.rhs)
lambda_y_0 = lambdify(eq_y_0.rhs)

## Move accelerations from accelerometers to origo:
P = Point("P")  # Point of the accelerometer, fixed in B with known velocity.
x_P, y_P, z_P = sp.symbols("x_P,y_P,z_P")
P.set_pos(O_B, x_P * B.x + y_P * B.y + z_P * B.z)

acceleration = P.acc(I).express(B).to_matrix(B)

u_acc = me.dynamicsymbols("u_acc")
v_acc = me.dynamicsymbols("v_acc")
subs = [
    (psi.diff().diff(), "r1d"),
    (psi.diff(), "r"),
    (psi, "psi"),
    (u.diff(), "u1d"),
    (v.diff(), "v1d"),
    (u, "u"),
    (v, "v"),
    (phi.diff().diff(), "p1d"),
    (phi.diff(), "p"),
    (phi, "phi"),
    (theta.diff().diff(), "q1d"),
    (theta.diff(), "q"),
    (theta, "theta"),
    (u_acc.diff(), "AccelX"),
    (v_acc.diff(), "AccelY"),
    (x_0, "x0"),
    (y_0, "y0"),
]

## Accelerometers
# Static gravitational force is also measured by the accelerometers:
gravity = -I.z*g
acceleration_g = (acceleration 
                  + gravity.express(B).to_matrix(B)      # Adding gravity projected on B
                  + sp.ImmutableDenseMatrix([0, 0, g]))  # Adding +g to make z acc zero at no heel.

u_,v_,r_ = sp.symbols('u,v,r')
phi_,theta_,psi_ = sp.symbols(r'\phi,\theta,\psi')

p1d,q1d = sp.symbols(r'\dot{p},\dot{q}')

# Removing dynamic symbols:
subs={
    phi.diff().diff(): p1d,
    theta.diff().diff(): q1d,
    psi.diff().diff(): r1d,

    phi.diff(): p,
    theta.diff(): q,
    psi.diff(): r,

    phi: phi_,
    theta: theta_,
    psi: psi_,
   
    
    u.diff():u1d,
    v.diff():v1d,
    
    u:u_,
    v:v_,
    r:r_,
}
expression = acceleration_g.subs(subs)
x2d_P,y2d_P,z2d_P = symbols(r"\ddot{x}_{P},\ddot{y}_{P},\ddot{z}_{P}")

# Accelerations in x,y,z direction at a point P (accelerometer)
eq_x2d_P = Eq(x2d_P, expression[0])
eq_y2d_P = Eq(y2d_P, expression[1])
eq_z2d_P = Eq(z2d_P, expression[2])

subs={
    u1d:'u1d',
    v1d:'v1d',
    r1d:'r1d',
    p1d:'p1d',
    q1d:'q1d',
    phi_:"phi",
    theta_:"theta",   
}

lambda_x2d_P = expression_to_python_method(eq_x2d_P.rhs.subs(subs), function_name='x2d_P')
lambda_y2d_P = expression_to_python_method(eq_y2d_P.rhs.subs(subs), function_name='y2d_P')
lambda_z2d_P = expression_to_python_method(eq_z2d_P.rhs.subs(subs), function_name='z2d_P')

## Simplified expression:
subs={
    phi.diff().diff(): 0,
    theta.diff().diff(): 0,
    psi.diff().diff(): r1d,

    phi.diff(): 0,
    theta.diff(): 0,
    psi.diff(): r,

    phi: phi_,
    theta: theta_,
    psi: psi_,
   
    
    u.diff():u1d,
    v.diff():v1d,
    
    u:u_,
    v:v_,
    r:r_,
}
expression_simplified = acceleration_g.subs(subs)

# Accelerations in x,y,z direction at a point P (accelerometer)
eq_x2d_P_simplified = Eq(x2d_P, expression_simplified[0])
eq_y2d_P_simplified = Eq(y2d_P, expression_simplified[1])
eq_z2d_P_simplified = Eq(z2d_P, expression_simplified[2])

subs={
    u1d:'u1d',
    v1d:'v1d',
    r1d:'r1d',
    p1d:'p1d',
    q1d:'q1d',
    phi_:"phi",
    theta_:"theta",   
}

lambda_x2d_P_simplified = expression_to_python_method(eq_x2d_P_simplified.rhs.subs(subs), function_name='x2d_P_simplified')
lambda_y2d_P_simplified = expression_to_python_method(eq_y2d_P_simplified.rhs.subs(subs), function_name='y2d_P_simplified')
lambda_z2d_P_simplified = expression_to_python_method(eq_z2d_P_simplified.rhs.subs(subs), function_name='z2d_P_simplified')


## Move accelerations from accelerometers to origo:
eqs = [
    eq_x2d_P,
    eq_y2d_P,
    eq_z2d_P,
]

#A, b = sp.linear_eq_to_matrix(eqs, [u1d, v1d, r1d])
#eq_accelerometer_to_origo = sp.Eq(sp.MutableDenseMatrix([u1d,v1d,r1d]), A.inv()*b)

eq_accelerometer_to_origo = sp.Eq(sp.MutableDenseMatrix([u1d,v1d,r1d]), sp.MutableDenseMatrix([
    sp.solve(eq_x2d_P_simplified,u1d)[0],
    sp.solve(eq_y2d_P_simplified,v1d)[0],
    sp.solve(eq_y2d_P_simplified,r1d)[0],
]))
  

subs={
    u1d:'u1d',
    v1d:'v1d',
    r1d:'r1d',
    p1d:'p1d',
    q1d:'q1d',
    phi_:"phi",
    theta_:"theta",
    x2d_P:"x2d_P",
    y2d_P:"y2d_P",
    z2d_P:"z2d_P",
}
lambda_u1d_from_accelerometer = expression_to_python_method(eq_accelerometer_to_origo.rhs[0].subs(subs), function_name='u1d_from_accelerometer')
lambda_v1d_from_accelerometer = expression_to_python_method(eq_accelerometer_to_origo.rhs[1].subs(subs), function_name='v1d_from_accelerometer')
lambda_r1d_from_accelerometer = expression_to_python_method(eq_accelerometer_to_origo.rhs[2].subs(subs), function_name='r1d_from_accelerometer')
