from bis_system import BisSystem
import seaman_symbol as ss

rho = ss.Bis('rho',description='Water density',unit = 'kg/m3',denominator=BisSystem.density)
g = ss.Bis('g',description='gravity',unit = 'm/s2',denominator=BisSystem.linear_acceleration)
L = ss.Bis('L',description='Perpendicular length',unit = 'm',denominator=BisSystem.length)
m = ss.Bis('m',description='Ship mass',unit = 'kg',denominator=BisSystem.mass)
volume = ss.Bis('disp',description='Ship volume',unit = 'm3',denominator=BisSystem.volume)
l_cg = ss.Bis('l_cg', description='Longitudinal centre of gravity', unit ='m', denominator=BisSystem.length)


t_a = ss.Bis('t_a',description='Draught aft',unit = 'm',denominator=BisSystem.length)
t_f = ss.Bis('t_f',description='Draught fore',unit = 'm',denominator=BisSystem.length)
u_w = ss.Bis('u_w',description='axial speed through water',unit='m/s',denominator=BisSystem.linear_velocity)
v_w = ss.Bis('v_w',description='transverse speed through water',unit = 'm/s',denominator=BisSystem.linear_velocity)
v_r = ss.Bis('v_r',description='transverse speed through water at rudder',unit = 'm/s',denominator=BisSystem.linear_velocity)

p = ss.Bis('p',description='roll velocity',unit = 'rad/s',denominator=BisSystem.angular_velocity)

r_w = ss.Bis('r_w',description='yaw rate through water',unit = 'rad/s',denominator=BisSystem.angular_velocity)
T = ss.Bis('T',description='Total thrust',unit = 'N',denominator=BisSystem.force)
T_prop = ss.Bis('T_prop',description='Thrust per propeller',unit = 'N',denominator=BisSystem.force)
n_prop = ss.Coefficient('n_prop', description='Number of propellers')

xx_rud = ss.Bis('xx_rud', description='Rudder x-coordinate', unit ='m', denominator=BisSystem.length)
zz_rud = ss.Bis('zz_rud', description='Rudder z-coordinate', unit ='m', denominator=BisSystem.length)
n_rud = ss.Coefficient('n_rud', description='Number of rudders')

X_vr = ss.Coefficient('X_vr')
X_vv = ss.Coefficient('X_vv')
X_rr = ss.Coefficient('X_rr')


Y_uv = ss.Coefficient('Y_uv')
Y_uuv = ss.Coefficient('Y_uuv')
Y_ur = ss.Coefficient('Y_ur')
Y_uur = ss.Coefficient('Y_uur')
C_d = ss.Coefficient('C_d')

K_ur  = ss.Coefficient('K_ur')
K_uur = ss.Coefficient('K_uur')
K_uv  = ss.Coefficient('K_uv')
K_uuv = ss.Coefficient('K_uuv')
K_up  = ss.Coefficient('K_up')
K_p   = ss.Coefficient('K_p')

K_vav   = ss.Coefficient('K_vav')
K_rar   = ss.Coefficient('K_rar')
K_pap   = ss.Coefficient('K_pap')

N_uv = ss.Coefficient('N_uv')
N_uuv = ss.Coefficient('N_uuv')
N_ur = ss.Coefficient('N_ur')
N_uur = ss.Coefficient('N_uur')

x_s = ss.Symbol('x_s', 'section x coordinate', unit ='m')
Cd_lever = ss.Symbol('Cd_lever', 'yaw moment lever to Cd', unit ='-')


Y_uudelta = ss.Coefficient('Y_uudelta','rudder speed dependent coefficient')
Y_Tdelta = ss.Coefficient('Y_Tdelta','rudder thrust dependent coefficient')
X_Yrdelta = ss.Coefficient('X_Yrdelta','rudder drag coefficient')
k_v = ss.Coefficient('k_v')
k_r = ss.Coefficient('k_r')
s = ss.Coefficient('s','rudder stall coefficient')

X  = ss.Bis('X',description='Total surge force',unit = 'N',denominator=BisSystem.force)
Y  = ss.Bis('Y',description='Total sway force',unit = 'N',denominator=BisSystem.force)
N  = ss.Bis('N',description='Total yawing moment',unit = 'Nm',denominator=BisSystem.moment)

X_h  = ss.Bis('X_h',description='surge force hull',unit = 'N',denominator=BisSystem.force)
X_res  = ss.Bis('X_res',description='calm water resistance',unit = 'N',denominator=BisSystem.force)

Y_h  = ss.Bis('Y_h',description='sway force hull',unit = 'N',denominator=BisSystem.force)
Y_v  = ss.Bis('Y_v',description='sway force due to drift',unit = 'N',denominator=BisSystem.force)
Y_r  = ss.Bis('Y_r',description='sway force due to yaw rate',unit = 'N',denominator=BisSystem.force)
Y_nl = ss.Bis('Y_nl',description='none linear sway force',unit = 'N',denominator=BisSystem.force)

N_h  = ss.Bis('N_h',description='yaw moment hull',unit = 'Nm',denominator=BisSystem.moment)
N_v  = ss.Bis('N_v',description='yaw moment due to drift',unit = 'Nm',denominator=BisSystem.moment)
N_r  = ss.Bis('N_r',description='yaw moment due to yaw rate',unit = 'Nm',denominator=BisSystem.moment)
N_nl = ss.Bis('N_nl',description='none linear yaw moment',unit = 'Nm',denominator=BisSystem.moment)

K  = ss.Bis('K',description='Total roll moment',unit = 'Nm',denominator=BisSystem.moment)
K_h  = ss.Bis('K_h',description='roll moment hull',unit = 'Nm',denominator=BisSystem.moment)

Y_rudder  = ss.Bis('Y_rudder',description='rudder side force',unit = 'N',denominator=BisSystem.force)
Y_rudder_u  = ss.Bis('Y_rudderu',description='rudder side force speed dependent part',unit = 'N',denominator=BisSystem.force)
Y_rudder_T  = ss.Bis('Y_rudderT',description='rudder side force thrust dependent part',unit = 'N',denominator=BisSystem.force)
delta  = ss.Bis('delta',description='rudder angle',unit = 'rad',denominator=BisSystem.angle)
delta_e  = ss.Bis('delta_e',description='effective rudder angle',unit = 'rad',denominator=BisSystem.angle)
X_rudder  = ss.Bis('X_rudder',description='rudder drag force',unit = 'N',denominator=BisSystem.force)
K_rudder  = ss.Bis('K_rudder',description='rudder roll moment',unit = 'Nm',denominator=BisSystem.moment)
N_rudder  = ss.Bis('N_rudder',description='rudder yawing moment',unit = 'Nm',denominator=BisSystem.moment)






