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
from scipy.linalg import svd


# Load Matlab data file and extract variables of interest
def CV_gauss(input_data, index_to_check):
    X = input_data

    # ros = RandomOverSampler(random_state=0)

    # X = ros.fit_sample(X)

    N, M = X.shape

    # Range of K's to try
    KRange = range(1, 8)
    T = len(KRange)

    covar_type = 'full'  # you can try out 'diag' as well
    reps = 5  # number of fits with different initalizations, best result will be kept

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

        BIC[t,] = gmm.bic(X)
        AIC[t,] = gmm.aic(X)

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

    print(CVE)
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


def draw_GMM(input_data):
    X, y = split_train_test(input_data, 9)
    y = np.argmax(input_data[:, 7:],1)

    U, S, V = svd(input_data[:,:], full_matrices=False)

    X = np.dot(input_data[:,:], V.T)

    # X = input_data
    N, M = X.shape

    # Number of clusters
    K = 5
    cov_type = 'full'
    # type of covariance, you can try out 'diag' as well
    reps = 10
    # number of fits with different initalizations, best result will be kept
    # Fit Gaussian mixture model
    gmm = GaussianMixture(n_components=K, covariance_type=cov_type, n_init=reps).fit(X)
    cls = gmm.predict(X)
    # extract cluster labels
    cds = gmm.means_
    # extract cluster centroids (means of gaussians)
    covs = gmm.covariances_
    # extract cluster shapes (covariances of gaussians)
    if cov_type == 'diag':
        new_covs = np.zeros([K, M, M])

    if cov_type == 'full':
        new_covs = np.zeros([K,M,M])

    count = 0
    for elem in covs:
        temp_m = np.zeros([M, M])
        for i in range(len(elem)):
            for j in range(len(elem)):
                temp_m[i][j] = elem[i][j]

        new_covs[count] = temp_m
        count += 1

    covs = new_covs

    print(cds)

    # Plot results:
    # figure(figsize=(14, 9))
    # clusterplot(X, clusterid=cls, centroids=cds, y=y, covars=covs)
    # show()


    ## In case the number of features != 2, then a subset of features most be plotted instead.
    figure(figsize=(14, 9))
    idx = [0, 1]  # feature index, choose two features to use as x and y axis in the plot
    clusterplot(X[:, idx], clusterid=cls, centroids=cds[:, idx], y=y, covars=covs[:, idx, :][:, :, idx])
    title('Clusterplot with GMM with origin')
    show()
