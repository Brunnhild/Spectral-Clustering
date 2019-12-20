import heapq
import numpy as np

class node:
    def __init__(self, index = -1):
        super().__init__()
        self.index = index
        self.adj = []
        self.parent = None
        self.key = float('inf')


    def __lt__(self, other):
        bol = self.key < other.key
        return bol
        

def mst(A):
    G = []
    for i in range(A.shape[0]):
        G.append(node(i))
    
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            if i == j:
                continue
            elif A[i][j] != float('inf'):
                G[i].adj.append(G[j])

    heap = G
    G = np.array(G)
    G[0].key = 0
    heapq.heapify(heap)

    while len(heap) > 0:
        u = heapq.heappop(heap)
        for v in u.adj:
            if A[u.index][v.index] >= v.key:
                continue
            flag = False
            for tmp in heap:
                if tmp.index == v.index:
                    flag = True
            if flag:
                v.parent = u
                v.key = A[u.index][v.index]
                heapq.heapify(heap)

    return G