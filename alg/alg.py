import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding
import matplotlib.pyplot as plt
import math
from alg.visualize import visual
from alg.score import get_cluster_score


def spectral_clus(train_data, sigma, k, v='LE', score=''):
    n = train_data.shape[0]
    W = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            W[i][j] = math.e ** (-(np.linalg.norm(train_data[i] - train_data[j])) / (2 * sigma ** 2))

    cat = SpectralClustering(k, affinity='precomputed').fit_predict(W)
    print(cat)
    get_cluster_score(train_data, W, cat, score)
    visual(train_data, W, cat, m=v)
    
