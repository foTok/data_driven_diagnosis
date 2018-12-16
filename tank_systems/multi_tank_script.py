import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
from tank_systems.components import multi_tank


mt = multi_tank(10, 10, 0.5)
mt.run(5, 5, 10, 10000, time_step=0.1, fault=('pip', 1, 0.3, 100))

trajectory = mt.trajectory()
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
mt.plot_trajectory(trajectory, 'pip', 0)
mt.plot_trajectory(trajectory, 'pip', 1)
mt.plot_trajectory(trajectory, 'pip', 2)
mt.plot_trajectory(trajectory, 'pip', 3)
mt.plot_trajectory(trajectory, 'pip', 4)
mt.plot_trajectory(trajectory, 'pip', 5)
mt.plot_trajectory(trajectory, 'pip', 6)
mt.plot_trajectory(trajectory, 'pip', 7)
mt.plot_trajectory(trajectory, 'pip', 8)
mt.plot_trajectory(trajectory, 'pip', 9)
mt.plot_qi()

print('DONE')
