"""
analysis data
"""

import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir) 
from data_manger.utilities import get_file_list
from bpsk_generator import Bpsk
import matplotlib.pyplot as pl
import numpy as np

#prepare data
PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
DATA_PATH = PATH + "\\data\\"
NORMAL_DATA_PATH = PATH + "\\data\\normal\\"

list_files = get_file_list(DATA_PATH)
for the_file in list_files:
    data_real = np.load(DATA_PATH+the_file)
    bpsk = Bpsk()
    data_pre = bpsk.generate_signal_with_input(0.0001, data_real[:, 0])
    np.save(NORMAL_DATA_PATH + the_file, data_pre)
    print("saved file {}.".format(the_file))
