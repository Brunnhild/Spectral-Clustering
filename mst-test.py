from MST import mst
from MST import node
import numpy as np
import heapq


if __name__ == '__main__':
    inf = float('inf')
    a = np.ones((9, 9))
    a *= inf
    for i in range(9):
        a[i][i] = 0

    a[0][1] = 4
    a[0][7] = 8
    a[1][2] = 8
    a[1][7] = 11
    a[2][3] = 7
    a[2][5] = 4
    a[2][8] = 2
    a[3][5] = 14
    a[4][5] = 10
    a[5][6] = 2
    a[6][7] = 1
    a[6][8] = 6
    a[7][8] = 7

    for i in range(9):
        for j in range(9):
            if i < j:
                a[j][i] = a[i][j]

    print(a)

    G = mst(a)
    for item in G:
        tmp = -1
        if item.parent != None:
            tmp = item.parent.index
        print('The node %d with parent %d and key %f' % (item.index, tmp, item.key))