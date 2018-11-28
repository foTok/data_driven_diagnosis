"""
The class defined to learn MTAN
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
from graph_model.utilities import graphviz_Bayes

class MTAN:
    """
    Learn a Tree Augmented Naive Bayesian Network.isinstance
    """
    #init
    def __init__(self):
        self.parameter_learner = Parameters_learning()
        self.mst_learner       = mst_learning()
        self.batch             = None
        self.MTAN              = Bayesian_network()
        self.n                 = 0
        self.cv                = False
        self.priori            = None
        self.res_pri           = None

    #interface
    def set_batch(self, batch):
        """
        Set the training data
        """
        self.parameter_learner.set_batch(batch)
        self.mst_learner.set_batch(batch[:, 6:])
        self.batch      = batch
        _, self.n       = batch.shape

    def set_priori(self, priori):
        """
        set priori knowledge
        """
        self.priori = priori

    def set_res_pri(self, priori):
        """
        RT
        """
        self.res_pri = priori

    def set_cv(self, cv):
        """
        Set contance varance flag
        """
        self.cv = cv

    def learn_MTAN(self):
        """
        Learn TAN
        """
        self.__struct()
        self.__parameter()

    def save_MTAN(self, file):
        """
        save TAN in a file
        """
        file = file + ".bn"
        s = pickle.dumps(self.MTAN)
        with open(file, "wb") as f:
            f.write(s)

    def save_graph(self, file):
        """
        save the TAN structure by graphviz
        """
        struct = self.MTAN.struct.struct
        graphviz_Bayes(struct, file + ".gv", self.n - 6 - 3)

    #inner functions
    def __struct(self):
        """
        Get the real struct
        """
        mst            = self.mst_learner.learn_mst(self.res_pri)
        struct         = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if self.priori[i, j] == 1:
                    struct[i, j] = 1
        for i in range(len(mst)):
            for j in range(len(mst)):
                if mst[i,j]==1:
                    struct[i+6, j+6] = 1
        self.MTAN.struct.set_struct(struct)
    
    def __parameter(self):
        """
        Learn the parameters
        """
        for fml in self.MTAN.struct:
            if len(fml)==1:
                continue
            para = self.parameter_learner.GGM_from_batch(fml, self.cv)
            self.MTAN.parameters.add_fml(fml, para)
