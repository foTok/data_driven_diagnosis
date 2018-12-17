import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import pickle
import numpy as np
from tank_systems.components import multi_tank
from data_manager2.data_cfg import data_cfg

# data file path
simu_id     = 0
data_path   = parentdir + '\\tank_systems\\data\\train{}\\'.format(simu_id)
if not os.path.isdir(data_path):
    os.makedirs(data_path)
prefix = 'multi-tank-simu'
# parameter cfg
n   = 10    # tank number
A   = 10    # tank cross sectional area
S   = 0.5   # pip cross sectional area
q   = 5     # input flow when input is on
h0  = 5     # control height 0
h1  = 10    # control height 1
time_step   = 1.0   # simulated time step
simu_time   = 1000  # simulated time
step_len    = int(simu_time/time_step)  # simulated time step length
fault   = ('tank', 0, 0.1, 200)
# the simulator
mt = multi_tank(n, A, S)

mt.run(q, h0, h1, step_len, time_step=time_step, fault=fault)
trajectory  = mt.trajectory()
mt.plot_trajectory(trajectory, 'tank', 0)
mt.plot_trajectory(trajectory, 'tank', 1)
mt.plot_trajectory(trajectory, 'tank', 2)
mt.plot_trajectory(trajectory, 'tank', 3)
mt.plot_trajectory(trajectory, 'tank', 4)
mt.plot_trajectory(trajectory, 'tank', 5)
mt.plot_trajectory(trajectory, 'tank', 6)
mt.plot_trajectory(trajectory, 'tank', 7)
mt.plot_trajectory(trajectory, 'tank', 8)
mt.plot_trajectory(trajectory, 'tank', 9)

print('DONE')
