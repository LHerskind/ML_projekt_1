# exercise 10.2.1
from scipy.io import loadmat
from toolbox_02450 import clusterplot
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler
from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim

def k_to_one(k_vector):
    return np.argmax(k_vector,1)


def Agglomerative(input_data, index_to_check):

    #ros = RandomOverSampler(random_state=0)

    #X, y = split_train_test(input_data, index_to_check)

    y = k_to_one(input_data[:,7:10])
    print(y)

    X = input_data

    #X = StandardScaler().fit_transform(X)

    #X, y = ros.fit_sample(X, y)

    U, S, V = svd(X, full_matrices=False)

    datamatrix_projected = np.dot(X, V[1:3].T)

    N, M = X.shape

    # Perform hierarchical/agglomerative clustering on data matrix
    Maxclust = 4
    Methods = ['average', 'complete', 'single']
    Metrics = ['seuclidean', 'mahalanobis']

    fignumber = 1
    for i in Methods:

        for j in Metrics:

            Method = i
            Metric = j

            Z = linkage(X, method=Method, metric=Metric)

            # Compute and display clusters by thresholding the dendrogram
            cls = fcluster(Z, criterion='maxclust', t=Maxclust)
            figure(fignumber)
            fignumber+=1
            clusterplot(datamatrix_projected, cls.reshape(cls.shape[0], 1), y=y)
            title(Method + ' ' + Metric)


            # Display dendrogram
            #max_display_levels = 6
            #figure(fignumber, figsize=(10, 4))
            #fignumber+=1
            #dendrogram(Z, truncate_mode='level', p=max_display_levels)


    show()

    print('Ran Exercise 10.2.1')


def split_train_test(input_matrix, index):
    y = input_matrix[:, index]
    X = np.delete(input_matrix, index, axis=1)
    print(X.shape)
    return X, y