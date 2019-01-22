'''
Using PCA to reduce features.
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import argparse
import numpy as np
from sklearn.decomposition import PCA

class PCA_feature_selection:
    def __init__(self, n_components=0.95):
        self.pca = PCA(n_components=n_components, svd_solver='full')

    def learn_from(self, X):
        reduced_X = self.pca.fit_transform(X)
        return reduced_X

    def transform(self, X):
        return self.pca.transform(X)
