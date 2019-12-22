import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding
import matplotlib.pyplot as plt
import math
from alg.visualize import visual
from alg.score import get_cluster_score
from alg.distance import get_distance


def spectral_clus(train_data, distance, sigma, k, v='LE', score=''):
    n = train_data.shape[0]
    
    (W, max_weight) = get_distance(train_data, distance)

    cat = SpectralClustering(k, affinity='precomputed').fit_predict(W)
    print(cat)
    get_cluster_score(train_data, W, cat, score)
    visual(train_data, W, cat, m=v)
    
