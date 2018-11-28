"""
some utilities
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import numpy as np
import matplotlib.pyplot as pl
from graphviz import Digraph
from queue import PriorityQueue
from graph_model.graph_component import Bayesian_structure

def number2vector(num, n):
    """
    convert a int number into a n bit vector
    """
    str_vec = bin(num).replace('0b','').zfill(n)
    vec = np.array([int(i) for i in str_vec])
    return vec

def vector2number(vec):
    """
    convert a vector to number
    """
    str_vec = [str(int(i)) for i in vec]
    str_bin_num = ''.join(str_vec)
    num = int(str_bin_num, 2)
    return num

def graphviz_Bayes(struct, file, fea_num):
    """
    convert struct into graphviz file
    """
    labels = ["F[0..6]"]
    for i in range(fea_num):
        labels.append("fe"+str(i))
    labels = labels + ["r0", "r1", "r2"]
    G = Digraph()
    #add nodes
    for i, lab in zip(range(len(labels)), labels):
        if i == 0:
            color = "yellow"
        elif  1<=i<1+fea_num:
            color = "green"
        else:
            color = "red"
        G.node(lab, lab, fillcolor=color, style="filled")
    #add edges from fault to features
    for i in range(fea_num):
        G.edge(labels[0], labels[i+1])
    #add edges from fault to residuals
    G.edge(labels[0], labels[-3], label="[0,1,5]")
    G.edge(labels[0], labels[-2], label="[2]")
    G.edge(labels[0], labels[-1], label="[4]")

    #add edges in struct
    #fault + feature + residual = 6 + fea_num (12) + 3
    #if node i is not a fault, the label should be i - 6 + 1, where
    #6 is the number of faults and 1 is the number of the fault label.
    n = len(struct)
    for i in range(6, n):
        for j in range(n):
            if struct[i, j] == 1:
                G.edge(labels[i-5], labels[j-5])
    print(G)
    G.render(file, view=True)
    print("Saved in: ", file)

def Guassian_cost(batch, fml, beta, var, norm):
    """
    the cost function
    """
    x = fml[:-1]
    y = fml[-1]
    X = batch[:, x]
    Y = batch[:, y]
    cost = Guassian_cost_core(X, Y, beta, var, norm)
    return cost

def Guassian_cost_core(X, Y, beta, var, norm):
    """
    cost function
    """
    var_basis = 1e-4
    N = len(Y)
    e = np.ones((N, 1))
    X = np.hstack((e, X))
    X = np.mat(X)
    Y_p = X * beta
    Y_p.shape = (N,)
    #number
    if isinstance(var, float) or isinstance(var, int):
        Var = var
    else:#vector
        Var = X * var
        Var = np.abs(np.array(Var))+ var_basis
        Var.shape = (N,)

    if norm:
        #cost0 = log(2*pi*var)/2
        cost0 = np.mean(np.log(2*np.pi*Var)) / 2
    else:
        cost0 = 0
    #cost1 = res**2/(2var)    
    cost1 = np.mean(np.array(Y_p - Y)**2 / (2*Var))
    cost = cost0 + cost1
    return cost

def priori_knowledge(fea_num = 12):
    """
    set priori knowledge
    """
    #0 ~can not determine
    #1 ~connection
    #-1~no connection
    #initially, all connection can not be determined.
    n = 6 + fea_num + 3
    pri_knowledge = np.zeros((n, n))
    #no self connection and downstair connection
    for i in range(n):
        for j in range(i+1):
            pri_knowledge[i,j] = -1
    #no connection between faults
    for i in range(6):
        for j in range(i+1, 6):
            pri_knowledge[i, j] = -1
            pri_knowledge[j, i] = -1
    #connections from faults to features or residuals
    #here, we first connect all faults to residuals.
    #but some edges will be deleted.
    for i in range(6):
        for j in range(6,n):
            pri_knowledge[i, j] = 1
    # model information
    #r0 --- node -3
    #unrelated fault [2,3,4]
    uf1 = [2,3,4]
    for i in uf1:
        pri_knowledge[i][-3] = -1
        pri_knowledge[-3][i] = -1
    #r1 --- node -2
    uf2 = [0,1,3,4,5]
    for i in uf2:
        pri_knowledge[i][-2] = -1
        pri_knowledge[-2][i] = -1
    #r3 --- node -1
    uf3 = [0,1,2,3,5]
    for i in uf3:
        pri_knowledge[i][-1] = -1
        pri_knowledge[-1][i] = -1
    #no connection between residuals
    for i in range(-3, 0):
        for j in range(-3, 0):
            pri_knowledge[i, j] = -1

    return pri_knowledge


def min_span_tree(MIEM, priori=None):
    """
    return a minimal span tree based on maximal information entropy matrix (MIEM)
    """
    queue = PriorityQueue()
    n = len(MIEM)
    #put all MIE into a priority queue
    for i in range(n):
        for j in range(i+1, n):
            queue.put((MIEM[i, j], (i, j)))
    #add edge one by one
    connection_block = []
    edges = []
    while len(edges) < n-1:
        _, edge = queue.get()
        if check_loop(connection_block, edge):
            if priori is None:
                edges.append(edge)
            else:
                i, j = edge
                if priori[i, j] != -1:
                    edges.append(edge)
    #undirected tree
    mst = und2d(edges)
    return mst

def check_loop(connection_block, edge):
    """
    check if could add the edge based on the connection blcok
    return True if could and add the edge into connection block
    return False if not
    """
    b0 = None
    b1 = None
    r  = True
    for b in connection_block:
        if edge[0] in b:
            b0 = b
        if edge[1] in b:
            b1 = b
        if (b0 is not None) and (b1 is not None):
            break
    if b0 == None:
        if b1 == None:
            connection_block.append([edge[0], edge[1]])
        else:
            b1.append(edge[0])
    else:
        if b1 == None:
            b0.append(edge[1])
        else:
            if b0 == b1:
                r = False
            else:
                i = connection_block.index(b1)
                del connection_block[i]
                for i in b1:
                    b0.append(i)
    return r

def und2d(edges):
    """
    tranfer an undirected graph into a directed graph
    """
    def _pop_connected_nodes(_i, _edges):
        _connected = set()
        _tmp_edges = set()
        for _edge in _edges:
            if _edge[0] == _i or _edge[1] == _i:
                _connected_node = _edge[0] if _edge[1] == _i else _edge[1]
                _connected.add(_connected_node)
                _tmp_edges.add(_edge)
        for _edge in _tmp_edges:
            _edges.remove(_edge)
        return _connected
    n = len(edges) + 1
    graph = np.zeros((n,n))
    first = edges[0][0]
    tail = set([first])
    while len(tail) != 0:
        tmp_tail = set()
        for i in tail:
            heads = _pop_connected_nodes(i, edges)
            for j in heads:
                graph[i, j] = 1
                tmp_tail.add(j)
        tail = tmp_tail     
    return graph

def und2od(edges, order):
    """
    Transfer an undirected graph into a directed graph based on order.
    edges: a list of edges
    order: a list
    """
    n = len(edges) + 1
    graph = np.zeros((n,n))
    for edge in edges:
        index0 = order.index(edge[0])
        index1 = order.index(edge[1])
        if index0 < index1:
            graph[index0, index1] = 1
        else:
            graph[index1, index0] = 1
    return graph

def compact_graphviz_struct(struct, file, labels):
    """
    graphviz the struct based on given labels and save the graph in file
    struct: a numpy array or a Bayesian_struct
    file: str
    lables: a list of tuples, [(label, color),...]
    """
    #make sure, struct is a numpy array
    if isinstance(struct, Bayesian_structure):
        struct = struct.struct
    n = len(struct)
    G = Digraph()
    #add nodes
    for i in range(n):
        if np.sum(struct[i, :]) + np.sum(struct[:, i]) > 0:
            label, color = labels[i]
            G.node(label, label, fillcolor=color, style="filled")
    #add edges
    for i in range(n):
        for j in range(n):
            if struct[i, j] == 1:
                G.edge(labels[i][0], labels[j][0])
    print(G)
    G.render(file, view=True)
    print("Saved in: ", file)

def read_normal_data(file_name, split_point=None, snr=None):
    """
    read normal data directly and add noise
    """
    #sig: time_step × feature
    sig = np.load(file_name)
    #add noise
    if snr is not None:
        NOISE_POWER_RATIO = 1/(10**(snr/10)) if snr is not None else 0
        signal_power = np.var(sig, 0)
        noise_power = NOISE_POWER_RATIO * signal_power
        noise_weight = noise_power**0.5
        #noise with Gaussian distribution
        noise = np.random.normal(0, 1, [len(sig), len(sig[0])]) * noise_weight
        sig = sig + noise
    split_point = int(len(sig) / 2) if split_point is None else split_point
    sig = sig[:split_point, :]
    return sig

def interval(start, inter, value):
    """
    return the interval number of value
    """
    i = 0
    start = start + inter
    while True:
        if value < start:
            break
        else:
            start = start + inter
            i     = i + 1
    return i

def discrete_data(data, discrete_num):
    """
    discrete data with same interval
    """
    #normalization
    std         = np.sqrt(np.var(data, axis=0))
    mean        = np.mean(data, axis=0)
    norm_data   = (data - mean)/std
    dis_data    = np.zeros(norm_data.shape)
    minmal      = np.min(data, axis=0)
    inter       = (np.max(data, axis=0) - minmal) / discrete_num
    assert (inter != 0).any()
    M, N = data.shape
    for i in range(N):
        for j in range(M):
            dis_data[j, i] = interval(minmal[i], inter[i], data[j, i])
    return dis_data
