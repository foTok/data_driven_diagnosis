'''
Chi-square Augmented Naive Bayesian Network
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import numpy as np
from graph_model.BN import BN
from graph_model.BN import Bayesian_adj
from graph_model.batch_statistic import CPT_BS
from graph_model.batch_statistic import Gauss_BS
from math import log
from math import exp

class C2AN:
    '''
    Chi-square Augmented Naive Bayesian Network
    '''
    def __init__(self, fault, obs):
        self.fault  = fault
        self.obs    = obs  
        self.n  = 1 + len(obs)
        self.statistic  = None
        self.mins   = None
        self.intervals  = None
        self.bins   = None
        self.adj    = None

    def _build_pmap_skeleton(self, d):
        '''
        Build-PMap-Skeleton.
        From [1] D. Koller and N. Friedman, Probabilistic Graphical Models, \
                 vol. 53, no. 9. Cambridge, Massachusetts, Londaon, England, 2013.
                 P85, Algorithm 3.3 Recovering the undirected skeleton for a distribution P that has a P-map
        Args:
            d: an int, bound on witness set.
        '''
        def witnesses(d, wit0, wit1, wit):
            '''
            Args:
                wit0: a list containing current collected variables.
                wit1: a list containing the rest of possible variables.
            '''
            if d==0 or not wit1:
                return
            for w in wit1:
                wit2 = wit0[:]
                wit3 = wit1[:]
                wit2.append(w)
                wit3.remove(w)
                wit.add(tuple(sorted(wit2)))
                witnesses(d-1, wit2, wit3, wit)

        def neighbors(xi, xj, adj):
            '''
            Returns:
                The neighbors of i or j
            '''
            nb = set()
            for i in range(len(adj)):
                if (adj[i, xi] == 1 or adj[i, xj] == 1) and \
                   (i != xi and i != xj):
                    nb.add(i)
            return list(nb)
        # Now the adj has two parts: the first part is f->obs, directed.
        # The second part is obs--obs, undirected. Learning algorithm 
        # will first handle the undirected part and then directs it. adj[i,i]=0
        self.adj    = np.array([[0]*self.n]*self.n)
        for i in range(self.n):
            for j in range(1, self.n):
                if i != j:
                    self.adj[i, j] = 1
        Uij = {}
        for i in range(1, self.n):
            for j in range(i+1, self.n):
                if self.adj[i, j] != 1: # No connection.
                    continue
                nb  = neighbors(i, j, self.adj)
                wit = set()
                wit.add(())
                witnesses(d, [], nb, wit)
                for u in wit:
                    if self.statistic.independent((i,), (j,), u):
                        Uij[(i,j)]  = u
                        self.adj[i, j]  = 0
                        self.adj[j, i]  = 0
                        break
        return Uij

    def _obs_undirected_nb(self, xj, adj=None):
        '''
        obtain the observed neighbors connected with an undirected edge
        Args:
            xj: node id
            adj: the adjacent matirx. None by default. In this case, adj=self.adj
        Returns:
            A set, the observed neghbors.
        '''
        onb = set()
        adj = self.adj if adj is None else adj
        for xi in range(1, self.n):
            if adj[xi, xj] == 1 and adj[xj, xi] == 1: # adj[xj, xj] = 0.
                onb.add(xi)
        return onb

    def _obs2(self, xj):
        '''
        The observed nodes to xj.
        Args:
            xj: the node
        Returns:
            a set.
        '''
        obs2 = set()
        for i in range(1, self.n):
            if self.adj[i, xj] == 1 and self.adj[xj, i] == 0:
                obs2.add(i)
        return obs2

    def _mark_immoralities(self, Uij):
        '''
        Mark-Immoralities.
        From [1] D. Koller and N. Friedman, Probabilistic Graphical Models, \
                 vol. 53, no. 9. Cambridge, Massachusetts, Londaon, England, 2013.
                 P86, Algorithm 3.4 Marking immoralities in the construction of a perfect map
        Args:
            Uij: Witnesses found by Build-PMap-Skeleton
        '''
        S = self.adj.copy()
        for xj in range(1, self.n):
            onb = self._obs_undirected_nb(xj, S)
            for xi in onb:
                for xk in onb:
                    if xi < xk and S[xi,xk]==0 and xj not in Uij[(xi, xk)]: # S is undireced graph (except fault node)
                        # Add the orientations Xi → Xj and Xj ← Xk to K
                        self.adj[xj, xi] = 0
                        self.adj[xj, xk] = 0

    def _R1(self, z):
        '''
        rule 1 for orienting edges.
        x -> y -- z ==> x -> y -> z
        Args:
            z: node id
        Returns:
            True if find a R1 subgraph
            False, otherwise.
        '''
        Y = self._obs_undirected_nb(z)
        for y in Y:
            X = self._obs2(y)
            for x in X:
                if self.adj[x, z]==0 and self.adj[z, x]==0:
                    # y->z, so adj[z, y] = 0
                    self.adj[z, y] = 0
                    return True
        return False

    def _R2(self, z):
        '''
        rule 2 for orienting edges.
        x -> y -> z and x -- z => x -> y -> z and x -> z
        Args:
            z: node id
        Returns:
            True if find a R2 subgraph
            False, otherwise.
        '''
        X0   = self._obs_undirected_nb(z)
        if not X0:
            return False
        Y   = self._obs2(z)
        for y in Y:
            X1  = self._obs2(y)
            X   = X0 & X1
            if X:
                x = X.pop()
                # x -> z. So, adj[z,x]=0
                self.adj[z,x]=0
                return True
        return False

    def _R3(self, z):
        '''
        rule 3 for orienting edges.
        y1 -- x -- y2 and y1 -> z <- y2 and x -- z => y1 -- x -- y2 and y1 -> z <- y2 and x -> z 
        Args:
            z: node id
        Returns:
            True if find a R1 subgraph
            False, otherwise.
        '''
        X  = self._obs_undirected_nb(z)    # for z
        if not X:
            return False
        Y   = self._obs2(z)
        if not Y:
            return False
        for y1 in Y:
            for y2 in Y:
                if y1!=y2:
                    for x in X:
                        if self.adj[x,y1]==1 and self.adj[y1,x]==1 \
                       and self.adj[x,y2]==1 and self.adj[y2,x]==1 \
                       and self.adj[y1,y2]==0 and self.adj[y2,y1]==0:
                            # x -> z. So, adj[z,x]=0
                            self.adj[z,x] = 0
                            return True
        return False

    def _Converge(self):
        '''
        Try to apply R1, R2 and R3 for each node
        '''
        for i in range(1, self.n):
            if self._R1(i):
                return False
            if self._R2(i):
                return False
            if self._R3(i):
                return False
        return True

    def _build_PDAG(self, d):
        '''
        Mark-Immoralities.
        From [1] D. Koller and N. Friedman, Probabilistic Graphical Models, \
                 vol. 53, no. 9. Cambridge, Massachusetts, Londaon, England, 2013.
                 P89, Algorithm 3.5 Finding the class PDAG characterizing the P-map of a distribution P
        Args:
            d: an int, bound on witness set.
        '''
        Uij = self._build_pmap_skeleton(d)
        self._mark_immoralities(Uij)
        while True:
            if self._Converge():
                break

    def set_batch(self, batch, mins, intervals, bins):
        '''
        Set batch with discretized parameters
        Args:
            batch: the data
            mins: the minimal values of variables
            intervals: the intervals of variables
            bins: the discretized numbers of variables
        '''
        self.mins   = np.array(mins)
        self.intervals  = np.array(intervals)
        self.bins   = np.array(bins)
        self.statistic  = CPT_BS(self.mins, self.intervals, self.bins)
        self.statistic.set_batch(batch)

    def Build_adj(self, d=3):
        '''
        Build adj
        Args:
            d: an int, boundness. (2+1(fault node))=3 by default.
        '''
        self._build_PDAG(d)
        # Orienting
        for i in range(1, self.n):
            for j in range(i+1, self.n):
                if self.adj[i,j]==1 and self.adj[j,i]==1: # i<j
                    self.adj[j, i] = 0

    def Build_BN(self):
        '''
        Build the BN
        '''
        bBN = BN(self.fault, self.obs)
        bBN.set_type('CPT', self.mins, self.intervals, self.bins)
        adj = Bayesian_adj(self.fault, self.obs)
        adj.set_adj(self.adj)
        bBN.set_adj(adj)
        for kid, parents in adj:
            vars = tuple(sorted(list(parents) + list(kid)))
            bBN.add_para(vars, self.statistic.para(vars))
            bBN.add_para(parents, self.statistic.para(parents))
        return bBN
