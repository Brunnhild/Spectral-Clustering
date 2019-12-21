import pandas as pd
import numpy as np

'''
Vaguely partitioned! 
Shrinkage to a line. 
'''
def get_data():
    train_data = np.array(pd.read_csv('data/glass.csv'))
    train_data = train_data[:, 1:-1]
    k = 6
    return (train_data, k)