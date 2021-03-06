import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import pickle
import numpy as np
from tank_systems.components import multi_tank
from data_manager2.data_cfg import data_cfg

# data file path
data_path   = parentdir + '\\tank_systems\\data\\test\\'
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
simu_time   = 2000  # simulated time
step_len    = int(simu_time/time_step)  # simulated time step length
# fault cfg
fault_cfg   = {'leakage': np.random.uniform(0.05, 0.2, 2), \
                'stuck': np.random.uniform(0.05, 0.2, 2)}
fault_time  = [int(i) for i in np.random.uniform(400, 1800, 6)]
# the simulator
mt = multi_tank(n, A, S)
# the data cfg
faults  = [0]*(2*n)
fault_paras = [0]*(2*n)
variables = [0]*(2*n+1)
variables[0] = 'qi'
for i in range(n):
    faults[i]   = 'tank_leakage{}'.format(i)
    faults[n+i] = 'pipe_stuck{}'.format(i)
    fault_paras[i]  = 'tank_leakage{}'.format(i)
    fault_paras[n+i]    = 'pipe_stuck{}'.format(i)
    variables[i+1]    = 'height{}'.format(i)
    variables[n+i+1]  = 'flow{}'.format(i)
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
                mt.run(q, h0, h1, step_len, time_step=time_step, fault=('tank' if f=='leakage' else 'pipe', i, p, ft))
                trajectory  = mt.np_trajectory()
                _file   = prefix+str(file_id)
                np.save(data_path+_file, trajectory)
                cfg.add_file(_file, \
                             fault_type='tank_leakage{}'.format(i) if f=='leakage' else 'pipe_stuck{}'.format(i), \
                             fault_time=ft, \
                             fault_para_name='tank_leakage{}'.format(i) if f=='leakage' else 'pipe_stuck{}'.format(i), \
                             fault_para_value=p)
                file_id += 1
                mt.reset()

cfg.save_cfg(data_path + 'cfg')

print('DONE')
