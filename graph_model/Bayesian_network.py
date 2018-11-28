"""
compose structure and parameters together
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir) 
from graph_model.graph_component import Bayesian_structure
from graph_model.graph_component import Bayesian_Gaussian_parameter

class Bayesian_network:
    """
    the class  to compose struct and parameters
    """
    def __init__(self):
        self.struct             = Bayesian_structure()
        self.parameters         = Bayesian_Gaussian_parameter()
