from alg import spectral_clus
import pandas as pd
import numpy as np

'''
Only two points after LLE.
'''
if __name__ == '__main__':
    train_data = np.array(pd.read_csv('data/wine.csv'))
    train_data = train_data[:, 1:]
    sigma = .5
    k = 3
    spectral_clus(train_data, sigma, k)