""""
learning Bayesian network by a fixed structure
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir) 
import torch
import numpy as np
import matplotlib.pyplot as pl
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list
from graph_model.graph_component import Bayesian_structure
from graph_model.graph_component import Bayesian_Gaussian_parameter
from graph_model.parameter_learning import Parameters_learning
from ddd.utilities import organise_data

#data amount
small_data = False
#settings
PATH = parentdir
DATA_PATH = PATH + "\\bpsk_navigate\\data\\test\\"
ANN_PATH = PATH + "\\ddd\\ann_model\\"
PGM_PATH = PATH + "\\graph_model\\pg_model\\"
fe_file = "FE0.pkl" if not small_data else "FE1.pkl"
struct_file = "linear_graph.npy"
step_len=100
epoch = 2000
batch = 2000

#load fe
FE = torch.load(ANN_PATH+fe_file)
FE.eval()

#load structure
np_struct = np.load(PGM_PATH + struct_file)

#prepare data
mana = BpskDataTank()
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=20, norm=True)

struct = Bayesian_structure()
struct.set_struct(np_struct)
struct.set_skip()
learning_module = Parameters_learning()
fml_tank = {}
for i in range(epoch):
    print("batch=", i)
    inputs, labels, _, res = mana.random_batch(batch, normal=0, single_fault=10, two_fault=0)
    feature = FE.fe(inputs)
    batch_data = organise_data(inputs, labels, res, feature)
    learning_module.reset()
    learning_module.set_batch(batch_data)
    for fml in struct:
        beta, var = learning_module.GGM_from_batch(fml)
        if i == 0:
            fml_tank[fml] = (beta, var)
        else:
            beta0, var0 = fml_tank[fml]
            beta = beta0 + (beta - beta0) / (i+1)
            var  = var0 + (var - var0) / (i+1)
            fml_tank[fml] = (beta, var)

print("learning end.")
