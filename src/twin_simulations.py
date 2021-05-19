import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from seaman.systems.composite.dynamic_ship import DynamicShip
from seaman.simulations.simulation_twin import TwinSimFakeProp,TwinSimFakePropRud
from seaman.simulations.speed_calibration import find_rev
from seaman.systems.composite.static_ship import StaticShip, StaticShipFakeProp
from seaman.helpers import ShipDict
import seaman.simulations as simulations
from mdl_helpers import mdl_motions, toolbox, froude_scaling
from seaman.systems.cppsystems.world import World



def preprocess_run(run, resample='1S'):
    df = run.df.copy()
    units = run.units.copy()
    
    meta_data = {
        'ScaleFactor' : run.model.scale_factor,
        'LOG' : run.loading_condition.lcg,
        'KG' : run.loading_condition.kg,
        'xm' : run.xm,
        'ym' : run.ym,
        'zm' : run.zm,
    }
    meta_data = pd.Series(meta_data)

    time_df, units = mdl_motions.add_ModelPos_motions(df=df, units = units, meta_data=meta_data)
    time_df['x0']-=time_df['x0'].iloc[0]
    time_df['y0']-=time_df['y0'].iloc[0]
    time_df['z0']-=time_df['z0'].iloc[0]
    
    #time_df.index = pd.TimedeltaIndex(time_df.index,unit='s')
    #time_df = froude_scaling.froude_scale(df=time_df, units=units, scale_factor=run.model.scale_factor, rho=1000)
    
    #time_df, units = mdl_motions.add_velocities(df=time_df, units=units, use_kalman_filter=False)
    #time_df['delta'] = time_df['Rudder/Angle']
    #units['delta'] = units['Rudder/Angle']
    
    #if not resample is None:
    #    time_df_resample = time_df.resample(resample).mean().dropna(subset=['x0'])
    #else:
    #    time_df_resample = time_df

    return time_df, units


def resimulate(time_df, shipdict, TwinSimClass=TwinSimFakePropRud):

    w = World()
    ship = setup_ship(shipdict=shipdict)
    w.register_unit(ship)
    
    simulation = TwinSimClass(ship = ship,time_df=time_df)
    
    simulation.run(max_simulation_time = 1000)
    
    return simulation

def plot_compare(simulation, time_df):
    
    fix, ax = plt.subplots()
    ax.axis('equal')
    
    time_df.plot(x='y0', y='x0', ax=ax, label='MDL')
        
    position_world = simulation.ship.res.position_world
    x = position_world[:,0]
    y = position_world[:,1]
    ax.plot(y, x, label='SIM')
    ax.grid(True)
    
    
    ax.legend()

def setup_ship(shipdict):
    
    #shipdict = ShipDict.load(ship_file_path)
    #ship = DynamicShip.from_shipdict(shipdict)

    static_ship = StaticShipFakeProp.from_shipdict(shipdict=shipdict)  # No propeller!
    ship = DynamicShip.from_static_ship(static_ship = static_ship)
    
    ship.cog.store('forces')

    for propeller in ship.propellers:
        propeller.store('thrust')
        propeller.store('torque')
        propeller.store('nfix')
        propeller.store('forces')
        
    for rudder in ship.rudders:    
        rudder.store('delta')
            
    return ship