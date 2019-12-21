import pandas as pd
import numpy as np

'''
Data points are very close.
'''
def get_data():
    train_data = np.array(pd.read_csv('data/balance-scale.csv'))
    trans = {'B': 0, 'L': 1, 'R': 2}
    for i in range(train_data.shape[0]):
        train_data[i][0] = trans[train_data[i][0]]
    train_data = train_data[:, :-1]
    k = 3
    return (train_data, k)