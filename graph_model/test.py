import pickle
import numpy as np
from BN import BN
from DBN import DBN

with open('graph_model\\pg_model\\train1\\20db\\0\\GSANmodel, d=2, ptype=CPT, ntype=D.bn', 'rb') as f:
    b = pickle.load(f)

c = b.logCost('normal', np.array([[1,1,1,1,10],[1,1,1,1,10],[1,1,1,1,10]]))

print('DONE')
