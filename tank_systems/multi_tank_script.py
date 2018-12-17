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
data_path   = parentdir + '\\tank_systems\\data\\{}\\'.format(simu_id)
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
time_step   = 0.1   # simulated time step
simu_time   = 1000  # simulated time
step_len    = int(simu_time/time_step)  # simulated time step length
# fault cfg
# fault_cfg   = {'leakage':[0.1, 0.2, 0.3, 0.4, 0.5], 'stuck':[0.1, 0.2, 0.3, 0.4, 0.5]}
# fault_time  = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 600, 800]
fault_cfg   = {'leakage':[0.1, 0.2], 'stuck':[0.1, 0.2]}
fault_time  = [100, 200]
# the simulator
mt = multi_tank(n, A, S)
# the data cfg
faults  = [0]*(2*n)
fault_paras = [0]*(2*n)
variables = [0]*(2*n)
for i in range(n):
    faults[i]   = 'tank_leakage{}'.format(i)
    faults[n+i] = 'pip_stuck{}'.format(i)
    fault_paras[i]  = 'tank_leakage{}'.format(i)
    fault_paras[n+i]    = 'pip_stuck{}'.format(i)
    variables[i]    = 'height{}'.format(i)
    variables[n+i]  = 'flow{}'.format(i)
cfg = data_cfg(variables, time_step, faults, fault_paras)


# simulate
file_id = 0
# normal
mt.run(q, h0, h1, step_len, time_step=time_step)
trajectory  = mt.np_trajectory()
_file   = prefix+str(file_id)
np.save(data_path+_file, trajectory)
cfg.add_file(_file)
file_id += 1
mt.reset()


# fault
for f in fault_cfg:
    for i in range(n):
        for p in fault_cfg[f]:
            for ft in fault_time:
                mt.run(q, h0, h1, step_len, time_step=time_step, fault=('tank' if f=='leakage' else 'pip', i, p, ft))
                trajectory = mt.np_trajectory()
                _file   = prefix+str(file_id)
                np.save(data_path+_file, trajectory)
                cfg.add_file(_file, \
                             fault_type='tank_leakage{}'.format(i) if f=='leakage' else 'pip_stuck{}'.format(i), \
                             fault_time=ft, \
                             fault_para_name='tank_leakage{}'.format(i) if f=='leakage' else 'pip_stuck{}'.format(i), \
                             fault_para_value=p)
                file_id += 1
                mt.reset()

cfg.save_cfg(data_path + 'cfg')

print('DONE')
