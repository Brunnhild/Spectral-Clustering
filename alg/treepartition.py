import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import math
from alg.MST import mst
from alg.MST import BFS
from alg.MST import make_tree
from alg.makegraph import make_graph
from alg.visualize import visual
from alg.distance import get_distance
from alg.score import get_cluster_score


def inspect(g):
    counts = {}
    for item in g:
        tmp = -1
        if item.parent != None:
            tmp = item.parent.index
        print('The node %d with parent %d and key %f and label %d' % (item.index, tmp, item.key, item.label))
        if item.label in counts:
            counts[item.label] += 1
        else:
            counts[item.label] = 1
    print(counts)


'''
The score of the current graph.
Target: Maximize the score of the cut.
'''
def get_score(W, g, cuts, max_weight, u, alpha):
    weight_score = W[u.parent.index][u.index] / max_weight

    counts = []
    total = len(g)
    for i in range(len(cuts) + 1):
        counts.append(0)
    for u in g:
        counts[u.label] += 1

    for i in range(len(counts)):
        counts[i] /= total
    counts = np.array(counts)
    partition = np.std(counts) / (math.sqrt(counts.shape[0] - 1) * np.mean(counts))
    sum = alpha * weight_score + (1 - alpha) * partition
    # print(sum_weights, partition)
    # sum += sum_weights

    return sum


def cut_tree(g, cuts):
    BFS(g[0], 0)
    for i in range(len(cuts)):
        BFS(cuts[i], i + 1)

def tree_partition(train_data, k, alpha=0.3, v='LE', graph=True, print_tree=False, distance_method='rbf', score=''):
    (W, max_weight) = get_distance(train_data, distance_method)

    g = mst(W)
    make_tree(g)
    # inspect(g)
    if graph:
        make_graph(W, g, [], max_weight, name='before')

    cuts = []

    for i in range(k - 1):
        min = float('inf')
        cutting = None
        for u in g:
            if u.parent == None:
                continue
            cuts.append(u)
            cut_tree(g, cuts)
            tmp = get_score(W, g, cuts, max_weight, u, alpha)
            cuts.pop()
            if tmp < min:
                min = tmp
                cutting = u
        cuts.append(cutting)
        cutting.parent = None
        print(cutting.index)

    cut_tree(g, cuts)
    res = []
    for u in g:
        res.append(u.label)
    res = np.array(res)

    if print_tree:
        inspect(g)
    get_cluster_score(train_data, W, res, score)

    if graph:
        make_graph(W, g, cuts, max_weight, name='after')

    visual(train_data, W, res, m=v)
