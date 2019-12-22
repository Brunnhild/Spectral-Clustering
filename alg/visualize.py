import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import SpectralEmbedding
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import NMF


def visual(D, W, res, m='LE'):
    if m == 'LE':
        fig = SpectralEmbedding(2, affinity='precomputed').fit_transform(W)
    elif m == 'PCA':
        fig = PCA(2).fit_transform(D)
    elif m == 'MDS':
        fig = MDS(2).fit_transform(D)
    elif m == 'Isomap':
        fig = Isomap(5, 2).fit_transform(D)
    elif m == 'SVD':
        fig = TruncatedSVD(2).fit_transform(D)
    elif m == 'KPCA':
        fig = KernelPCA(2).fit_transform(D)
    elif m == 'NMF':
        fig = NMF(2).fit_transform(D)
    else:
        return
    color = np.array(['red', 'green', 'blue', 'purple', 'pink', 'orange'])
    plt.scatter(fig[:, 0], fig[:, 1], color=color[res])
    plt.show()
