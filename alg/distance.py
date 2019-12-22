import numpy as np
import math


def distance_rbf(train_data):
    n = train_data.shape[0]
    W = np.zeros((n, n))
    sigma = .5

    for i in range(n):
        for j in range(n):
            W[i][j] = math.e**(
                -(np.linalg.norm(train_data[i] - train_data[j])) /
                (2 * sigma**2))

    return W


def distance_cos(train_data):
    n = train_data.shape[0]
    W = np.zeros((n, n))
    sigma = .5

    for i in range(n):
        for j in range(n):
            W[i][j] = train_data[i] @ train_data[j].T / (np.linalg.norm(
                train_data[i], ord=1) * np.linalg.norm(train_data[j], ord=1))

    return W


def distance_md(train_data):
    n = train_data.shape[0]
    cov = np.cov(train_data.T)
    W = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            d = train_data[i] - train_data[j]
            W[i][j] = math.sqrt(d @ np.linalg.inv(cov) @ d.T)

    return W


def gauss_weighted_distance(train_data):
    n = train_data.shape[0]
    cov = np.cov(train_data.T)
    W = np.zeros((n, n))
    sigma = .5

    for i in range(n):
        for j in range(n):
            d = train_data[i] - train_data[j]
            W[i][j] = math.e ** (-(d @ np.linalg.inv(cov) @ d.T) / 2)

    return W


def get_distance(train_data, m='rbf'):
    train_data = train_data.astype(np.float32)
    n = train_data.shape[0]
    max_weight = -float('inf')
    if m == 'rbf':
        W = distance_rbf(train_data)
    elif m == 'cos':
        W = distance_cos(train_data)
    elif m == 'md':
        W = distance_md(train_data)
    elif m == 'gwd':
        W = gauss_weighted_distance(train_data)

    for i in range(n):
        for j in range(n):
            if W[i][j] > max_weight:
                max_weight = W[i][j]

    return (W, max_weight)
