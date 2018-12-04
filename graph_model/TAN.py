"""
The class defined to learn TAN
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import pickle
import numpy as np
from graph_model.BN import BN
from graph_model.min_span_tree import mst_learning

class TAN:
    '''
    Learn a Tree Augmented Naive Bayesian Network.isinstance
    '''
    #init
    def __init__(self):
        pass

    #interface
    def set_batch(self, batch):
        pass

    def save_TAN(self, file):
        pass
