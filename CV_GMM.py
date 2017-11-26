from matplotlib.pyplot import figure, plot, legend, xlabel, show
import numpy as np
from scipy.io import loadmat
from sklearn.mixture import GaussianMixture
from sklearn import model_selection
from imblearn.over_sampling import RandomOverSampler
from toolbox_02450 import clusterplot
import numpy as np
from matplotlib.pyplot import figure, subplot, plot, hist, title, show
from sklearn.mixture import GaussianMixture


# Load Matlab data file and extract variables of interest
def CV_gauss(input_data, index_to_check):
    X, y = split_train_test(input_data, index_to_check)

    ros = RandomOverSampler(random_state=0)

    X, y = ros.fit_sample(X, y)

    N, M = X.shape

    # Range of K's to try
    KRange = range(1, 15)
    T = len(KRange)

    covar_type = 'full'  # you can try out 'diag' as well
    reps = 10  # number of fits with different initalizations, best result will be kept

    # Allocate variables
    BIC = np.zeros((T,))
    AIC = np.zeros((T,))
    CVE = np.zeros((T,))

    # K-fold crossvalidation
    CV = model_selection.KFold(n_splits=10, shuffle=True)

    for t, K in enumerate(KRange):
        print('Fitting model for K={0}'.format(K))

        # Fit Gaussian mixture model
        gmm = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps).fit(X)

        BIC[t, ] = gmm.bic(X)
        AIC[t, ] = gmm.aic(X)

        # For each crossvalidation fold
        for train_index, test_index in CV.split(X):
            # extract training and test set for current CV fold
            X_train = X[train_index]
            X_test = X[test_index]

            # Fit Gaussian mixture model to X_train
            gmm = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps).fit(X_train)

            # compute negative log likelihood of X_test
            CVE[t] += -gmm.score_samples(X_test).sum()

            # Plot results

    figure(1)
    plot(KRange, BIC, '-*b')
    plot(KRange, AIC, '-xr')
    plot(KRange, 2 * CVE, '-ok')
    legend(['BIC', 'AIC', 'Crossvalidation'])
    xlabel('K')

    show()


def split_train_test(input_matrix, index):
    y = input_matrix[:, index]
    X = np.delete(input_matrix, index, axis=1)
    print(X.shape)
    return X, y
