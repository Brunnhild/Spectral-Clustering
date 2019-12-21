import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding
import matplotlib.pyplot as plt
import math


def spectral_clus(train_data, sigma, k):
    n = train_data.shape[0]
    W = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            W[i][j] = math.e ** (-(np.linalg.norm(train_data[i] - train_data[j])) / (2 * sigma ** 2))

    cat = SpectralClustering(k, affinity='precomputed').fit_predict(W)
    res = SpectralEmbedding(2, affinity='precomputed').fit_transform(W)
    print(cat)
    color = np.array(['red', 'green', 'blue', 'purple', 'pink', 'orange'])
    plt.scatter(res[:, 0], res[:, 1], color=color[cat])
    plt.show()
