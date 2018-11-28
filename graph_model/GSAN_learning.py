"""
train a DAG Bayesian network
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import torch
import pickle
import numpy as np
import matplotlib.pyplot as pl
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list
from graph_model.Bayesian_learning import Bayesian_structure
from graph_model.Bayesian_learning import Bayesian_learning
from graph_model.utilities import priori_knowledge
from graph_model.utilities import graphviz_Bayes
from ddd.utilities import organise_data

#data amount
small_data = True
#priori knowledge
pri_knowledge = priori_knowledge()

#settings
PATH = parentdir
DATA_PATH = PATH + "\\bpsk_navigate\\data\\" + ("big_data\\" if not small_data else "small_data\\")
ANN_PATH = PATH + "\\ddd\\ann_model\\" + ("big_data\\" if not small_data else "small_data\\")
PGM_PATH = PATH + "\\graph_model\\pg_model\\" + ("big_data\\" if not small_data else "small_data\\")
fe_file = "FE0.pkl"
struct_file = "GSAN0.gv"
pgm_file = "GSAN0.bn"
fea_num = 12
step_len=100
epoch = 1000
batch = 2000 if not small_data else 1000

#load fe
FE = torch.load(ANN_PATH+fe_file)
FE.eval()

#prepare data
mana = BpskDataTank()
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=20, norm=True)

#labels, 6; features, fea_num; residuals 3.
BL = Bayesian_learning(6+fea_num+3)
# BL.set_cv(True)
BL.set_priori(pri_knowledge)
for i in range(epoch):
    inputs, labels, _, res = mana.random_batch(batch, normal=0.2, single_fault=10, two_fault=0)
    feature = FE.fe(inputs)
    batch_data = organise_data(inputs, labels, res, feature)
    BL.set_batch(batch_data)
    BL.step(i)

best, _ = BL.best_candidate()
graphviz_Bayes(best.struct, PGM_PATH + struct_file, fea_num)

BN = BL.best_BN()
s = pickle.dumps(BN)
with open(PGM_PATH + pgm_file, "wb") as f:
    f.write(s)
