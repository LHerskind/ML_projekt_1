# exercise 11.4.1
import numpy as np
from matplotlib.pyplot import (figure, imshow, bar, title, xticks, yticks, cm,
                               subplot, show)
from scipy.io import loadmat
from toolbox_02450 import gausKernelDensity
from sklearn.neighbors import NearestNeighbors


def Outlier(input_data, index_to_check):
    X, y = split_train_test(input_data, index_to_check)

    N, M = np.shape(X)

    # Restrict the data to images of "2"

    ### Gausian Kernel density estimator
    # cross-validate kernel width by leave-one-out-cross-validation
    # (efficient implementation in gausKernelDensity function)
    # evaluate for range of kernel widths
    widths = X.var(axis=0).max() * (2.0 ** np.arange(-10, 3))
    logP = np.zeros(np.size(widths))
    for i, w in enumerate(widths):
        print('Fold {:2d}, w={:f}'.format(i, w))
        density, log_density = gausKernelDensity(X, w)
        logP[i] = log_density.sum()

    val = logP.max()
    ind = logP.argmax()

    width = widths[ind]
    print('Optimal estimated width is: {0}'.format(width))

    # evaluate density for estimated width
    density, log_density = gausKernelDensity(X, width)

    # Sort the densities
    i = (density.argsort(axis=0)).ravel()
    density = density[i].reshape(-1, )
    print('The index of the lowest GKD estimator object: {0}'.format(i[0:5]))
    print('The value of the lowest GKD estimator object: {0}'.format(density[0:5]))

    # Plot density estimate of outlier score
    figure(1)
    bar(range(20), density[:20])
    title('Density estimate')

    # Plot possible outliers

    ### K-neighbors density estimator
    # Neighbor to use:
    K = 5

    # Find the k nearest neighbors
    knn = NearestNeighbors(n_neighbors=K).fit(X)
    D, i = knn.kneighbors(X)

    density = 1. / (D.sum(axis=1) / K)

    # Sort the scores
    i = density.argsort()
    density = density[i]
    print('The index of the lowest KNN 5 neighbours density object: {0}'.format(i[0:5]))
    print('The value of the lowest KNN 5 neighbours density object: {0}'.format(density[0:5]))

    # Plot k-neighbor estimate of outlier score (distances)
    figure(3)
    bar(range(20), density[:20])
    title('KNN density: Outlier score')
    # Plot possible outliers

    ### K-nearest neigbor average relative density
    # Compute the average relative density

    knn = NearestNeighbors(n_neighbors=K).fit(X)
    D, i = knn.kneighbors(X)
    density = 1. / (D.sum(axis=1) / K)
    avg_rel_density = density / (density[i[:, 1:]].sum(axis=1) / K)

    # Sort the avg.rel.densities
    i_avg_rel = avg_rel_density.argsort()
    avg_rel_density = avg_rel_density[i_avg_rel]

    print('The index of the lowest KNN average relative density object: {0}'.format(i_avg_rel[0:5]))
    print('The value of the lowest KNN average relative density object: {0}'.format(avg_rel_density[0:5]))

    # Plot k-neighbor estimate of outlier score (distances)
    figure(5)
    bar(range(20), avg_rel_density[:20])
    title('KNN average relative density: Outlier score')

    # Plot possible outliers
    ### Distance to 5'th nearest neighbor outlier score
    K = 5

    # Find the k nearest neighbors
    knn = NearestNeighbors(n_neighbors=K).fit(X)
    D, i = knn.kneighbors(X)

    # Outlier score
    score = D[:, K - 1]
    # Sort the scores
    i = score.argsort()
    score = score[i[::-1]]
    print('The index of the highest KNN 5 neighbours outlier score: {0}'.format(i[0:5]))
    print('The value of the highest KNN 5 neighbours outlier score: {0}'.format(score[0:5]))

    # Plot k-neighbor estimate of outlier score (distances)
    figure(7)
    bar(range(20), score[:20])
    title('5th neighbor distance: Outlier score')
    # Plot possible outliers

    show()

    print('Ran Exercise 11.4.1')


def split_train_test(input_matrix, index):
    y = input_matrix[:, index]
    X = np.delete(input_matrix, index, axis=1)
    print(X.shape)
    return X, y
