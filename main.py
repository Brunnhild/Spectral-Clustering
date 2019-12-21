from data.iris import get_data
from alg.treepartition import tree_partition


if __name__ == '__main__':
    train_data, k = get_data()
    tree_partition(train_data, k)
