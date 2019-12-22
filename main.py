from data.seeds import get_data
from alg.treepartition import tree_partition
from alg.alg import spectral_clus


if __name__ == '__main__':
    # score = 'cs'
    # alpha = .3
    # train_data, k = get_data()
    # tree_partition(train_data, k, alpha=alpha, v='KPCA', graph=False, print_tree=False, distance_method='rbf', score=score)
    # tree_partition(train_data, k, alpha=alpha, v='KPCA', graph=True, print_tree=False, distance_method='cos', score=score)
    # spectral_clus(train_data, sigma=.5, k=k, v='none', score=score)

    score = 'dbi'
    alpha = .3
    train_data, k = get_data()
    spectral_clus(train_data, distance='rbf', sigma=.5, k=k, v='none', score=score)
    spectral_clus(train_data, distance='gwd', sigma=.5, k=k, v='none', score=score)
