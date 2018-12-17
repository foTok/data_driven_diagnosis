import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import pickle
from tank_systems.components import multi_tank
from data_manager2.data_cfg import data_cfg

# data file path
simu_id     = 1
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
fault_cfg   = {'leakage':[0.1, 0.2, 0.3, 0.4, 0.5], 'stuck':[0.1, 0.2, 0.3, 0.4, 0.5]}
fault_time  = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 600, 800]
# the simulator
mt = multi_tank(n, A, S)
# the data cfg
faults  = []
fault_paras = []
for i in range(n):
    faults.append('tank_leakage{}'.format(i))
    faults.append('pip_stuck{}'.format(i))
    fault_paras.append('tank_leakage{}'.format(i))
    fault_paras.append('pip_stuck{}'.format(i))
cfg = data_cfg(faults, fault_paras)

# save function
def save_traj(_file, trajectory):
    _s = pickle.dumps(trajectory)
    with open(data_path + _file, "wb") as f:
        f.write(_s)

# simulate
file_id = 0
# normal
mt.run(q, h0, h1, step_len, time_step=time_step)
trajectory  = mt.trajectory()
_file   = prefix+str(file_id)
save_traj(_file, trajectory)
cfg.add_file(_file)
file_id += 1
mt.reset()

# this block for debug, commented when running
# mt.plot_trajectory(trajectory, 'tank', 0)
# mt.plot_trajectory(trajectory, 'tank', 5)
# mt.plot_trajectory(trajectory, 'tank', 9)
# mt.plot_trajectory(trajectory, 'pip', 0)
# mt.plot_trajectory(trajectory, 'pip', 5)
# mt.plot_trajectory(trajectory, 'pip', 9)

# fault
for f in fault_cfg:
    for i in range(n):
        for p in fault_cfg[f]:
            for ft in fault_time:
                mt.run(q, h0, h1, step_len, time_step=time_step, fault=('tank' if f=='leakage' else 'pip', i, p, ft))
                trajectory = mt.trajectory()
                _file   = prefix+str(file_id)
                save_traj(_file, trajectory)
                cfg.add_file(_file, \
                             fault_type='tank_leakage{}'.format(i) if f=='leakage' else 'pip_stuck{}'.format(i), \
                             fault_time=ft, \
                             fault_para_name='tank_leakage{}'.format(i) if f=='leakage' else 'pip_stuck{}'.format(i), \
                             fault_para_value=p)
                file_id += 1
                mt.reset()
                
                # this block for debug, commented when running
                # mt.plot_trajectory(trajectory, 'tank', 0)
                # mt.plot_trajectory(trajectory, 'tank', 5)
                # mt.plot_trajectory(trajectory, 'tank', 9)
                # mt.plot_trajectory(trajectory, 'pip', 0)
                # mt.plot_trajectory(trajectory, 'pip', 5)
                # mt.plot_trajectory(trajectory, 'pip', 9)

cfg.save_cfg(data_path + 'cfg')

print('DONE')
