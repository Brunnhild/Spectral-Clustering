import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding
import matplotlib.pyplot as plt
import math
from MST import mst
from MST import BFS
from MST import make_tree
from makegraph import make_graph


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
def get_score(W, g, cuts, max_weight, u):
    alpha = 0.3
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


if __name__ == '__main__':
    train_data = np.array(pd.read_csv('data/iris.csv'))
    train_data = train_data[:, :-1]
    # Number of classes to divide.
    k = 3

    sigma = .5

    n = train_data.shape[0]
    W = np.zeros((n, n))
    max_weight = -float('inf')

    for i in range(n):
        for j in range(n):
            W[i][j] = math.e ** (-(np.linalg.norm(train_data[i] - train_data[j])) / (2 * sigma ** 2))
            if W[i][j] > max_weight:
                max_weight = W[i][j]

    fig = SpectralEmbedding(2, affinity='precomputed').fit_transform(W)

    g = mst(W)
    make_tree(g)
    # inspect(g)
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
            tmp = get_score(W, g, cuts, max_weight, u)
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

    inspect(g)
    make_graph(W, g, cuts, max_weight, name='after')

    color = np.array(['red', 'green', 'blue', 'purple', 'pink', 'orange'])
    plt.scatter(fig[:, 0], fig[:, 1], color=color[res])
    plt.show()
