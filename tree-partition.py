import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding
import matplotlib.pyplot as plt
import math
from MST import mst
from MST import BFS
from MST import make_tree


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
def get_score(W, g, cuts):
    sum_weights = 0
    for u in g:
        if u.parent == None:
            continue
        sum_weights += W[u.index][u.parent.index]
    for u in cuts:
        if u.parent == None:
            continue
        sum_weights -= W[u.parent.index][u.index]

    counts = []
    total = len(g)
    for i in range(len(cuts) + 1):
        counts.append(0)
    for u in g:
        counts[u.label] += 1

    partition = 0
    for item in counts:
        partition += total / item
    sum = sum_weights - partition
    # print(sum_weights, partition)
    # sum += sum_weights

    return sum



def cut_tree(g, cuts):
    BFS(g[0], 0)
    for i in range(len(cuts)):
        BFS(cuts[i], i + 1)


if __name__ == '__main__':
    train_data = np.array(pd.read_csv('data/balance-scale.csv'))
    trans = {'B': 0, 'L': 1, 'R': 2}
    for i in range(train_data.shape[0]):
        train_data[i][0] = trans[train_data[i][0]]
    train_data = train_data[:, :-1]

    sigma = .5

    n = train_data.shape[0]
    W = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            W[i][j] = math.e ** (-(np.linalg.norm(train_data[i] - train_data[j])) / (2 * sigma ** 2))

    fig = SpectralEmbedding(2, affinity='precomputed').fit_transform(W)

    g = mst(W)
    make_tree(g)
    # inspect(g)

    # Number of classes to divide.
    k = 3
    cuts = []

    for i in range(k - 1):
        max = -float('inf')
        cutting = None
        for u in g:
            if u.parent == None:
                continue
            cuts.append(u)
            cut_tree(g, cuts)
            tmp = get_score(W, g, cuts)
            cuts.pop()
            if tmp > max:
                max = tmp
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

    color = np.array(['red', 'green', 'blue', 'purple', 'pink', 'orange'])
    plt.scatter(fig[:, 0], fig[:, 1], color=color[res])
    plt.show()
