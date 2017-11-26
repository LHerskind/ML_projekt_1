# exercise 10.2.1
from matplotlib.pyplot import figure, show
from scipy.io import loadmat
from toolbox_02450 import clusterplot
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import numpy as np


def Agglomerative(input_data, index_to_check):

    X, y = split_train_test(input_data, index_to_check)

    N, M = X.shape

    # Perform hierarchical/agglomerative clustering on data matrix
    Method = 'single'
    Metric = 'euclidean'

    Z = linkage(X, method=Method, metric=Metric)

    # Compute and display clusters by thresholding the dendrogram
    Maxclust = 4
    cls = fcluster(Z, criterion='maxclust', t=Maxclust)
    figure(1)
    clusterplot(X, cls.reshape(cls.shape[0],1), y=y)

    # Display dendrogram
    max_display_levels=6
    figure(2,figsize=(10,4))
    dendrogram(Z, truncate_mode='level', p=max_display_levels)

    show()

    print('Ran Exercise 10.2.1')


def split_train_test(input_matrix, index):
    y = input_matrix[:, index]
    X = np.delete(input_matrix, index, axis=1)
    print(X.shape)
    return X, y