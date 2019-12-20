import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding
import matplotlib.pyplot as plt
import math
from MST import mst


if __name__ == '__main__':
    train_data = np.array(pd.read_csv('data/iris.csv'))
    train_data = train_data[:, :-1]
    sigma = .5

    n = train_data.shape[0]
    W = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            W[i][j] = math.e ** (-(np.linalg.norm(train_data[i] - train_data[j])) / (2 * sigma ** 2))

    g = mst(W)

    for item in g:
        tmp = -1
        if item.parent != None:
            tmp = item.parent.index
        print('The node %d with parent %d and key %f' % (item.index, tmp, item.key))