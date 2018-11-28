"""
The class defined to learn TAN
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import pickle
import numpy as np
from graph_model.Bayesian_network import Bayesian_network
from graph_model.parameter_learning import Parameters_learning
from graph_model.min_span_tree import mst_learning
from graph_model.utilities import compact_graphviz_struct

class TAN:
    """
    Learn a Tree Augmented Naive Bayesian Network.isinstance
    """
    #init
    def __init__(self):
        self.parameter_learner = Parameters_learning()
        self.mst_learner       = mst_learning()
        self.batch             = None
        self.TAN               = Bayesian_network()
        self.n                 = 0
        self.cv                = False

    #interface
    def set_batch(self, batch):
        """
        Set the training data
        """
        self.parameter_learner.set_batch(batch)
        self.mst_learner.set_batch(batch[:, 1:])
        self.batch      = batch
        _, self.n       = batch.shape

    def set_cv(self, cv):
        """
        Set contance varance flag
        """
        self.cv = cv

    def learn_TAN(self):
        """
        Learn TAN
        """
        self.__struct()
        self.__parameter()

    def save_TAN(self, file):
        """
        save TAN in a file
        """
        file = file + ".bn"
        s = pickle.dumps(self.TAN)
        with open(file, "wb") as f:
            f.write(s)

    def save_graph(self, file, labels):
        """
        save the TAN structure by graphviz
        """
        struct = self.TAN.struct.struct
        compact_graphviz_struct(struct, file + ".gv", labels)

    #inner functions
    def __struct(self):
        """
        Get the real struct
        """
        mst            = self.mst_learner.learn_mst()
        struct         = np.zeros((self.n, self.n))
        struct[0, 1:]  = 1
        for i in range(len(mst)):
            for j in range(len(mst)):
                if mst[i,j]==1:
                    struct[i+1, j+1] = 1
        self.TAN.struct.set_struct(struct)
    
    def __parameter(self):
        """
        Learn the parameters
        """
        for fml in self.TAN.struct:
            if len(fml)==1:
                continue
            para = self.parameter_learner.GGM_from_batch(fml, self.cv)
            self.TAN.parameters.add_fml(fml, para)
