import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from seaman.systems.composite.dynamic_ship import DynamicShip
from seaman.simulations.simulation_twin import TwinSimFakeProp,TwinSimFakePropRud
from seaman.simulations.speed_calibration import find_rev
from seaman.systems.composite.static_ship import StaticShip, StaticShipFakeProp
from seaman.helpers import ShipDict
import seaman.simulations as simulations
from seaman.systems.cppsystems.world import World

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