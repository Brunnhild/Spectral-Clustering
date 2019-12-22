from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score


def get_cluster_score(train_data, W, res, m=''):
    if m == '':
        return
    elif m == 'ss':
        # big
        print('The score is %f' % (silhouette_score(W, res, metric='precomputed')))
    elif m == 'cs':
        # big
        print('The score is %f' % (calinski_harabasz_score(train_data, res)))
    elif m == 'dbi':
        # small
        print('The score is %f' % (davies_bouldin_score(train_data, res)))