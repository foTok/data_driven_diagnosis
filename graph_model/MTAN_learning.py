"""
Learning the TAN classifiers for 6 faults independently and a merged TAN classifier
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import torch
import pickle
import numpy as np
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list
from graph_model.min_span_tree import mst_learning
from graph_model.utilities import priori_knowledge
from graph_model.utilities import compact_graphviz_struct
from graph_model.MTAN import MTAN
from ddd.utilities import organise_data

#data amount
small_data = True

#settings
PATH = parentdir
DATA_PATH = PATH + "\\bpsk_navigate\\data\\" + ("big_data\\" if not small_data else "small_data\\")
ANN_PATH = PATH + "\\ddd\\ann_model\\" + ("big_data\\" if not small_data else "small_data\\")
PGM_PATH = PATH + "\\graph_model\\pg_model\\" + ("big_data\\" if not small_data else "small_data\\")
fe_file = "FE0.pkl"
step_len=100
batch = 20000

#prepare data
mana = BpskDataTank()
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=20, norm=True)
#load fe
FE = torch.load(ANN_PATH+fe_file)
FE.eval()
#sample data
inputs, labels, _, res = mana.random_batch(batch, normal=0.4, single_fault=10, two_fault=0)
feature = FE.fe(inputs)
batch_data = organise_data(inputs, labels, res, feature)

#priori knowledge
_, n = batch_data.shape
fea_num = n - 6 - 3
pri_knowledge = priori_knowledge(fea_num)

#priori that focuses on residuals
res_priori = np.zeros((n-6, n-6))
res_priori[-3:,-3:] = -1

#train
learner     = MTAN()
learner.set_batch(batch_data)
learner.set_priori(pri_knowledge)
#adding res_priori will cause infinite loop
#learner.set_res_pri(res_priori)
learner.learn_MTAN()
#save
file_name = "MTAN"
learner.save_graph(PGM_PATH + file_name)
learner.save_MTAN(PGM_PATH + file_name)

print("Done")
