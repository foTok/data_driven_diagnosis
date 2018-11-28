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
from graph_model.TAN import TAN
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
_, fea_num = batch_data.shape
fea_num = fea_num - 6 - 3
pri_knowledge = priori_knowledge(fea_num)
#labels
fault_labels = []
for i in range(6):
    fault_labels.append(("f"+str(i), "yellow"))
fea_labels = []
for i in range(fea_num):
    fea_labels.append(("fe"+str(i), "green"))
res_labels = []
for i in range(3):
    res_labels.append(("r"+str(i), "red"))
labels = fault_labels + fea_labels + res_labels

for i in range(6):
    #data index
    index       = pri_knowledge[i, :]
    index       = (index==1)
    index[i]    = True
    #labels
    label_i     = [lab for lab, index_i in zip(labels, index) if index_i]
    #learn
    batch_i     = batch_data[:, index]
    learner     = TAN()
    learner.set_batch(batch_i)
    learner.learn_TAN()
    #save
    file_name = "TAN" + str(i)
    learner.save_graph(PGM_PATH + file_name, label_i)
    learner.save_TAN(PGM_PATH + file_name)

print("Done")
