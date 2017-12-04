from matplotlib.pyplot import figure, title, plot, ylim, legend, show
import numpy as np
from toolbox_02450 import clusterval
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler


def split_train_test(input_matrix, index):
    y = input_matrix[:, index]
    X = np.delete(input_matrix, index, axis=1)
    print(X.shape)
    return X, y


def Evaluate(input_data, index_to_check):
    X = input_data[:,:7]

    y = np.argmax(input_data[:,7:10], 1)
    # X = StandardScaler().fit_transform(X)

    N, M = np.shape(X)

    split_index = int(X.shape[0] * 0.5)
    print(split_index)
    X_train = X[:split_index, :]
    X_test = X[split_index:, :]
    y_test = y[split_index:]

    # Maximum number of clusters:
    K = 10

    # Allocate variables:
    Rand = np.zeros((K,))
    Jaccard = np.zeros((K,))
    NMI = np.zeros((K,))

    for k in range(K):
        cls = GaussianMixture(n_components=K, covariance_type="full", n_init=10).fit(X)
        Rand[k], Jaccard[k], NMI[k] = clusterval(y.ravel(), cls.predict(X))
        print(Rand[k], Jaccard[k], NMI[k])

    # Plot results:

    figure(1)
    title('Cluster validity ')
    plot(np.arange(K) + 1, Rand)
    plot(np.arange(K) + 1, Jaccard)
    plot(np.arange(K) + 1, NMI)
    ylim(-2, 1.1)
    legend(['Rand', 'Jaccard', 'NMI'], loc=4)
    show()

    print('Ran Exercise 10.1.3')
