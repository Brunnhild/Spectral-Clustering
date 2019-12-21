import pandas as pd
import numpy as np

'''
Only two points after LLE.
'''
def get_data():
    train_data = np.array(pd.read_csv('data/wine.csv'))
    train_data = train_data[:, 1:]
    k = 3
    return (train_data, k)