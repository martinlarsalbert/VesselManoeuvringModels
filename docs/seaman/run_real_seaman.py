import pandas as pd
import numpy as np
import quantities as pq

from seaman.systems.composite.static_ship import StaticShip
from seaman.systems.cppsystems.world import World

def calculate_static_ship_df(df, shipdict, no_prop=True):
    
    s = StaticShip.from_shipdict(shipdict, no_prop=no_prop)
    assert isinstance(s, StaticShip)
    
    w = World()
    w.register_unit(s.cog)
    
    if no_prop:
        assert not hasattr(s.cog.bl, 'fixprop_0')

    result = df.apply(func=calculate_static_ship, s=s, axis=1, no_prop=no_prop)
    return result

def calculate_static_ship(row, s, no_prop=True):
    u = row.get('u_w',0)
    v = row.get('v_w',0)
    r = row.get('r_w',0)

    w = 0
    p = 0
    q = 0


    shipdict=s.shipdict

    s.position_world = [0, 0, 0]
    s.attitude_world = [0, 0, 0]  # (Roll must be 0 due to static ship stability)

    s.linear_velocity = [u, v, w]
    s.angular_velocity = [p, q, r]

    delta = row['delta']
    s.delta = pq.Quantity(delta,pq.rad)

    n_rudders = len(shipdict.rudder_particulars)
    thrust = row['T_prop'] # Thrust per propeller!!!

    if n_rudders == 0:
        pass
    elif n_rudders == 1:
        s.cog.bl.rudder_0.inputs.thrust = thrust # Thrust per propeller!!!
    elif n_rudders == 2:
        s.cog.bl.rudder_0.inputs.thrust = thrust # Thrust per propeller!!!
        s.cog.bl.rudder_1.inputs.thrust = thrust # Thrust per propeller!!!
    elif n_rudders == 3:
        s.cog.bl.rudder_0.inputs.thrust = thrust # Thrust per propeller!!!
        s.cog.bl.rudder_1.inputs.thrust = thrust # Thrust per propeller!!!
        s.cog.bl.rudder_2.inputs.thrust = thrust # Thrust per propeller!!!
    
    else:
        raise ValueError('Cannot simulate with more than 3 rudders')

    s.cog.initialize()
    s.evaluate()

    # ToDo: Where is really the origo of the forces? BL or CG or..?
    #force = s.cog.bl.outputs.forces  # This is moment around lpp/2 NOT cg!
    force = s.cog.outputs.forces  # This is force and moment at [lpp/2 + lcg,0,wl]!
    force = move_forces_to_lpp2(shipdict=shipdict, force=force)

    output = pd.Series()
    output['fx'] = force[0]

    if no_prop:
        output['fx']+=thrust

    output['fy'] = force[1]  # /(shipdict.design_particulars['disp']*1025*9.81)
    output['fz'] = force[2]
    output['mx'] = force[3]
    output['my'] = force[4]
    output['mz'] = force[5]

    rudder_forces = np.zeros(6)
    for rudder in s.rudders:
        rudder_force = np.array(rudder.outputs.forces)
        rudder_forces+=rudder_force

    rudder_forces = move_forces_to_lpp2(shipdict=shipdict, force=rudder_forces)
    output['fx_rudders_seaman'] = rudder_forces[0]
    output['fy_rudders'] = rudder_forces[1] #NOTE this is same notation as in notebooks!!
    output['fz_rudders_seaman'] = rudder_forces[2]
    output['mx_rudders_seaman'] = rudder_forces[3]
    output['my_rudders_seaman'] = rudder_forces[4]
    output['mz_rudders_seaman'] = rudder_forces[5]

    output['delta_e'] = s.rudders[0].outputs.delta_e

    return output

def move_forces_to_lpp2(shipdict,force):

    # Moving the forces to: [Lpp/2,0,WL] --> dv = [-lcg,0,0]
    lcg = shipdict.design_particulars['lcg']
    kg = shipdict.design_particulars['kg']
    ta = shipdict.design_particulars['ta']
    tf = shipdict.design_particulars['tf']
    tm = (ta + tf) / 2

    wl_to_vcg = kg - tm

    moved_force = np.array(force)

    fy = force[1]
    moved_force[5] += lcg * fy
    moved_force[3] += wl_to_vcg * fy
    return moved_force



def update_ship_parameters(parameters, input_map, s):
    counter = 0
    for input_group in input_map:
        input_adresses = input_group[0]
        values = input_group[1]

        for parameter_name in values:
            for input_adress in input_adresses:
                parameter_value = parameters[counter]
                set_parameter(input_adress, parameter_name, parameter_value, s=s)

            counter += 1

def set_parameter(input_adress, parameter_name, parameter_value, s):

    if parameter_name=='xxrud':
        set_parameter_name = 'position_world'
        position=getattr(input_adress, set_parameter_name)
        lcg=-s.cog.bl.inputs.position_world[0]
        position[0]=parameter_value+lcg
        set_value = position

    elif parameter_name=='yyrud':
        set_parameter_name = 'position_world'
        position=getattr(input_adress, set_parameter_name)
        position[1]=parameter_value
        set_value = position

    elif parameter_name=='zzrud':
        set_parameter_name = 'position_world'
        position=getattr(input_adress, set_parameter_name)
        position[2]=parameter_value
        set_value = position

    else:
        set_parameter_name = parameter_name
        set_value = parameter_value

    setattr(input_adress, set_parameter_name, set_value)

def get_parameter(input_adress, parameter_name, s):

    if parameter_name=='xxrud':
        position=getattr(input_adress, 'position_world')
        lcg=-s.cog.bl.inputs.position_world[0]
        return position[0]+lcg
    elif parameter_name=='yyrud':
        position=getattr(input_adress, 'position_world')
        return position[1]
    elif parameter_name=='zzrud':
        position=getattr(input_adress, 'position_world')
        return position[2]
    else:
        return getattr(input_adress, parameter_name)


def estimator(parameters, X, s, input_map, no_prop=True):
    update_ship_parameters(parameters=parameters, input_map=input_map, s=s)
    results = X.apply(func=calculate_static_ship, s=s, axis=1, no_prop=no_prop)

    return results