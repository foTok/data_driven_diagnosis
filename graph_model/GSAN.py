'''
Greedy Search Augmented Naive Bayesian Network
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import numpy as np
from graph_model.BN import BN
from graph_model.BN import Bayesian_adj
from graph_model.DBN import DBN
from graph_model.DBN import DBN_adj
from graph_model.batch_statistic import CPT_BS
from graph_model.batch_statistic import Gauss_BS
from math import log
from math import exp

class GSAN:
    '''
    Greedy search augmented naive Beyesian Network
    '''
    def __init__(self, fault, obs):
        self.queue  = {}
        self.type   = None
        self.ntype  = None
        self.statistic  = None
        self.mins   = None
        self.intervals  = None
        self.bins   = None
        self.delay  = 0
        self.fault  = fault
        self.obs    = obs  
        self.n  = None
        self.vars_r_cost_cache  = {}
        self.graph_r_cost_cache = {}

    def init_queue(self):
        '''
        init the search queue
        '''
        if self.ntype == 'S':
            adj = Bayesian_adj(self.fault, self.obs)
        elif self.ntype == 'D':
            adj = DBN_adj(self.fault, self.obs)
        else:
            raise RuntimeError('Unknown type.')
        adj.naive_init()
        cost = float('inf')
        self.queue[adj] = cost

    def best_candidate(self):
        '''
        get the best candidate
        '''
        queue = self.queue
        graph, cost = min(queue.items(), key=lambda x:x[1])
        return graph, cost

    def set_ntype(self, _type):
        '''
        '''
        if _type == 'S':# Static BN
            self.n  = len(self.obs) + 1
        elif _type == 'D': # Dynamic BN
            self.n  = 2*len(self.obs) + 1
        else:
            raise RuntimeError('unknown parameter type')
        self.ntype = _type

    def set_type(self, _type, mins=None, intervals=None, bins=None):
        '''
        Set the parameter type.
        '''
        if _type == 'CPT':
            assert mins is not None and intervals is not None and bins is not None
            self.mins   = np.array(mins)
            self.intervals  = np.array(intervals)
            self.bins   = np.array(bins)
            self.statistic  = CPT_BS(self.mins, self.intervals, self.bins)
        elif _type == 'GAU':
            self.statistic  = Gauss_BS()
        else:
            raise RuntimeError('unknown parameter type')
        self.type   = _type

    def l_cost(self, adj):
        '''
        compute the likelihood cost
        '''
        cost = 0
        for kid, parents in adj:
            cost_u = self.statistic.fml_cost(kid, parents)
            cost += cost_u
        return cost

    def r_cost(self, adj):
        '''
        compute the regular cost for graph
        '''
        if adj in self.graph_r_cost_cache:
            return self.graph_r_cost_cache[adj]
        #Now we know the regular cost is not computed for the adj
        cost = 0
        for kid, parents in adj:
            vars = tuple(sorted(list(parents) + list(kid)))
            if vars in self.vars_r_cost_cache:
                cost_u = self.vars_r_cost_cache[vars]
            else:#Now we know the fml is not cached
                if self.type == 'CPT':
                    p_bins = self.bins[list(parents)]
                    k_bin = self.bins[list(kid)]
                    cost_u = p_bins.prod()*(k_bin[0]-1)
                elif self.type == "GAU":
                    cost_u = len(parents)+1
                    cost_u = exp(cost_u)
                else:
                    raise RuntimeError('Unknown type.')
                self.vars_r_cost_cache[vars] = cost_u
            cost += cost_u
        return cost

    def cost(self, adj):
        '''
        compute the cost of a structure
        '''
        l_cost = self.l_cost(adj)
        r_cost = self.r_cost(adj)
        cost = l_cost + self.delay * r_cost
        return cost

    def step(self, batch):
        '''
        step forward
        Args:
            batch: the data
            time_step: to indicate time step
        Returns:
            _cost: _the current best cost
        '''
        if not self.queue:
            print("Empty queue. Break")
            return
        best, _cost = self.best_candidate()
        #self.queue.clear()
        self.statistic.set_batch(batch)
        n = len(batch)
        self.delay = log(n)/(2*n)
        # Update best
        cost = self.cost(best)
        self.queue[best] = cost
        #change randomly
        for i in range(self.n):
            for j in range(self.n):
                if not best.available(i, j):
                    continue
                # add
                if best.add_perm(i, j):
                    add_best = best.clone()
                    add_best.add_edge(i, j)
                    add_cost = self.cost(add_best)
                    self.queue[add_best] = add_cost
                # remove
                if best.remove_perm(i, j):
                    rem_best = best.clone()
                    rem_best.remove_edge(i, j)
                    rem_cost = self.cost(rem_best)
                    self.queue[rem_best] = rem_cost
                # reverse
                if best.reverse_perm(i, j):
                    rev_best = best.clone()
                    rev_best.reverse_edge(i, j)
                    rev_cost = self.cost(rev_best)
                    self.queue[rev_best] = rev_cost
        return _cost

    def best_BN(self):
        '''
        best BN
        '''
        best_adj, _ = self.best_candidate()
        bBN = BN(self.fault, self.obs) if self.ntype == 'S' else DBN(self.fault, self.obs)
        bBN.set_type(self.type, self.mins, self.intervals, self.bins)
        bBN.set_adj(best_adj.adj())
        for kid, parents in best_adj:
            if self.type == 'CPT':
                vars = tuple(sorted(list(parents) + list(kid)))
                bBN.add_para(vars, self.statistic.para(vars))
                bBN.add_para(parents, self.statistic.para(parents))
            elif self.type == 'GAU':
                fml = tuple(list(parents) + list(kid))
                bBN.add_para(fml, self.statistic.para(fml))
            else:
                raise RuntimeError('Unknown type')
        return bBN
