import numpy as np
from graphviz import render

def sub_data(input, label, select):
    index = (label==select)
    data = input[index,:,:]
    return data

def best_index(scores, thresh):
    thresh = thresh**2
    index = np.argsort(-scores)
    I = 0
    features = []
    for i in index:
        I += (scores[i]**2)
        features.append(i)
        if I > thresh:
            break
    return features

def simple_net(important_vars, important_features, file, var_list=None, mode_list=None, dir='TB'):
    file = file if file.endswith('.gv') else (file+'.gv')
    dot_bg = ['digraph G {\nrankdir = '+dir+';\nsubgraph {']
    dot_ed = ['} /* closing subgraph */\n}']
    nodes = []
    edges = []

    features = set()
    rank0 = '{rank = same;'
    for mode in important_features:
        mode_name = 'm{}'.format(mode) if mode_list is None else mode_list[mode]
        rank0 += mode_name+';'
        nodes.append('{} [label=\"{}\", shape=box]'.format(mode_name, mode_name))   # mode nodes
        for fea in important_features[mode]:
            features.add(fea)
            edges.append('fe{} -> {}'.format(fea, mode_name)) # feature to mode
    rank0 += '}'

    vars = set()
    rank1 = '{rank = same;'
    for fea in features:
        nodes.append('fe{} [label=\"fe{}\", shape=ellipse]'.format(fea, fea))   # feature nodes
        rank1 += 'fe{};'.format(fea)
        for v in important_vars[fea]:
            var_name = 'v{}'.format(v) if var_list is None else var_list[v]
            vars.add(v)
            edges.append('{} -> fe{}'.format(var_name, fea)) # variable to feature
    rank1 += '}'

    rank2 = '{rank = same;'
    for v in vars:
        var_name = 'v{}'.format(v) if var_list is None else var_list[v]
        nodes.append('{} [label=\"{}\", shape=ellipse]'.format(var_name, var_name))   # var nodes
        rank2 += (var_name+';')
    rank2 += '}'
    dot_file = dot_bg + nodes + edges + [rank0, rank1, rank2] + dot_ed
    with open(file, 'w') as f:
        f.write('\n'.join(dot_file))
    render('dot', 'svg', file)
