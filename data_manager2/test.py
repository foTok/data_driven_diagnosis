import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
from data_manager2.data_manager import mt_data_manager

data_path = parentdir + '\\tank_systems\\data\\{}\\'.format(0)

mt_mana = mt_data_manager()
mt_mana.load_data(data_path)
mt_mana.add_noise(20)
data = mt_mana.random_h_batch(210, 10, 0.05, 1.0)

print('Done')