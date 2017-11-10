import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.linalg import svd
# from sklearn import model_selection
import sklearn.linear_model as lm
from mpl_toolkits.mplot3d import Axes3D
from sklearn import cross_validation, tree
from matplotlib import pyplot as plt
from toolbox_02450 import feature_selector_lr, bmplot
from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim
from scipy import stats
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import sklearn.neural_network as nn
import neurolab as nl
import CV_BestK
import CV_bestnn_classification
import CV_big_shit
import CV_bestnn_reg

# from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show

attributeNames = ['MPG', 'Cylinders', 'Displacment', 'Horsepower', 'Weight (lbs)', 'Acceleration (MPH)', 'Model year', 'Origin']
origins = ['USA', 'Europe', 'Japan']


def load_from_file(file):
    datamatrix = np.loadtxt(file)
    datamatrix[:, 0] = datamatrix[:, 0] * -1
    print(datamatrix.shape)
    return datamatrix


def create_plots(datamatrix, datamatrix_std):
    create_boxplots(datamatrix)
    create_histo(datamatrix)
    correlation_plots(datamatrix, datamatrix_std)


def create_boxplots(datamatrix):
    plt.figure(figsize=(2, 4))
    for i in range(0, 8):
        plt.subplot(2, 4, i + 1)
        plt.boxplot(datamatrix[:, i])
        plt.ylabel(attributeNames[i])
    plt.show()


def create_histo(datamatrix):
    plt.figure(figsize=(2, 4))

    for i, color in enumerate(['red', 'yellow', 'blue', 'brown', 'green', 'cyan', 'purple', 'orange'], start=0):
        plt.subplot(2, 4, i + 1)
        plt.hist(datamatrix[:, i], color=color, edgecolor='black')
        plt.xlabel(attributeNames[i])

    plt.show()


def summary_statistics(datamatrix):
    for i in range(0, 8):
        mean_x = datamatrix[:, i].mean()
        std_x = datamatrix[:, i].std(ddof=1)
        median_x = np.median(datamatrix[:, i])
        range_x = datamatrix[:, i].max() - datamatrix[:, i].min()
        print(attributeNames[i])
        print('Mean:', mean_x)
        print('Standard Deviation:', std_x)
        print('Median:', median_x)
        print('Range:', range_x)
        print('Min', min_x)
        print('Max', max_x)
        print('First quantile', twentyfive_x)
        print('Third quantile', seventyfive_x)


def convert_using_1_to_k(inputmatrix):
    return np.hstack((inputmatrix[:, :7], np.reshape(get_one_to_k_matrix(), (len(datamatrix), 3))))


def get_one_to_k_matrix():
    return [vectorized(e) for e in datamatrix[:, 7]]


def vectorized(j):
    e = np.zeros((3, 1))
    e[int(j) - 1] = 1.0 / np.sqrt(3)
    return e


def std_cov_coff_matrices(datamatrix):
    datamatrix_std = StandardScaler().fit_transform(datamatrix)
    cov_matrix = np.cov(datamatrix_std.T)
    coff_matrix = np.corrcoef(datamatrix_std.T)
    return datamatrix_std, cov_matrix, coff_matrix


def correlation_plots(datamatrix, datamatrix_std):
    plt.figure(figsize=(8, 8))

    for m1 in range(len(attributeNames)):
        for m2 in range(len(attributeNames)):
            plt.subplot(len(attributeNames), len(attributeNames), m1 * len(attributeNames) + m2 + 1)
            for c in range(0, 3):
                class_mask = datamatrix_std[:, 7 + c].ravel() > 0
                plt.plot(datamatrix[class_mask, m2], datamatrix[class_mask, m1], '.', label=origins[c])
                if m1 == len(attributeNames) - 1:
                    plt.xlabel(attributeNames[m2])
                else:
                    plt.xticks([])
                if m2 == 0:
                    plt.ylabel(attributeNames[m1])
                else:
                    plt.yticks([])
    plt.legend()
    plt.show()


def svd_graph(datamatrix_std, made1_to_k):
    U, S, V = svd(datamatrix_std, full_matrices=False)
    rho = (S * S) / (S * S).sum()
    rho_cummulative = np.cumsum(rho)

    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(6, 4))
        plt.bar(range(1, len(rho) + 1), rho, alpha=0.6, align='center', label='Individual explained variance')
        plt.step(range(1, len(rho) + 1), rho_cummulative, where='mid', label='Cumulative explained variance')
        plt.ylabel("Explained variance ratio")
        plt.xlabel('Principal components')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

    datamatrix_projected = np.dot(datamatrix_std, V.T)

    if is3D:
        f = plt.figure()
        ax = f.add_subplot(111, projection='3d')
        for c in range(0, 3):
            class_mask = datamatrix_std[:, 7 + c].ravel() > 0
            ax.scatter(datamatrix_projected[class_mask, 0], datamatrix_projected[class_mask, 1], datamatrix_projected[class_mask, 2], label=origins[c])
        ax.view_init(45, 45)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.title('Origins show across PC1 & PC2 & PC3')
        plt.show()
    else:
        for c in range(0, 3):
            class_mask = datamatrix_std[:, 7 + c].ravel() > 0
            plt.scatter(datamatrix_projected[class_mask, 0], datamatrix_projected[class_mask, 1], label=origins[c])
        plt.legend()
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Origins show across PC1 & PC2')
        plt.show()


def split_train_test(input_matrix, index):
    y = input_matrix[:, index];
    X = np.delete(input_matrix, index, axis=1)
    print(X.shape)
    return X, y


def linear_reg(input_matrix, index, outer_cross_number, inner_cross_number):
    X, y = split_train_test(input_matrix, index)
    N, M = X.shape
    K = outer_cross_number
    # CV = model_selection.KFold(K,True)

    neurons = 20
    learning_goal = 0.02
    max_epochs = 500
    show_error_freq = 100

    temp = attributeNames[index]
    attributeNamesShorter = attributeNames
    attributeNamesShorter.remove(temp)
    CV = cross_validation.KFold(N, K, shuffle=True)

    Features = np.zeros((M, K))
    Error_train = np.empty((K, 1))
    Error_test = np.empty((K, 1))
    Error_train_fs = np.empty((K, 1))
    Error_test_fs = np.empty((K, 1))
    Error_train_nofeatures = np.empty((K, 1))
    Error_test_nofeatures = np.empty((K, 1))
    Error_train_nn = np.empty((K, 1))
    Error_test_nn = np.empty((K, 1))
    k = 0
    for train_index, test_index in CV:
        X_train = X[train_index, :]
        y_train = y[train_index]
        X_test = X[test_index, :]
        y_test = y[test_index]
        internal_cross_validation = inner_cross_number

        Error_train_nofeatures[k] = np.square(y_train - y_train.mean()).sum() / y_train.shape[0]
        Error_test_nofeatures[k] = np.square(y_test - y_test.mean()).sum() / y_test.shape[0]

        m = lm.LinearRegression(fit_intercept=True).fit(X_train, y_train)
        Error_train[k] = np.square(y_train - m.predict(X_train)).sum() / y_train.shape[0]
        Error_test[k] = np.square(y_test - m.predict(X_test)).sum() / y_test.shape[0]
        textout = ''
        selected_features, features_record, loss_record = feature_selector_lr(X_train, y_train, internal_cross_validation,
                                                                              display=textout)

        Features[selected_features, k] = 1
        # .. alternatively you could use module sklearn.feature_selection
        if len(selected_features) is 0:
            print('No features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).')
        else:
            m = lm.LinearRegression(fit_intercept=True).fit(X_train[:, selected_features], y_train)
            Error_train_fs[k] = np.square(y_train - m.predict(X_train[:, selected_features])).sum() / y_train.shape[0]
            Error_test_fs[k] = np.square(y_test - m.predict(X_test[:, selected_features])).sum() / y_test.shape[0]

            ann = nl.net.newff([[-3, 3]] * M, [neurons, 1], [nl.trans.TanSig(), nl.trans.PureLin()])
            y_train_2 = [[x] for x in y_train]
            ann.train(X_train, y_train_2, goal=learning_goal, epochs=max_epochs, show=show_error_freq)
            y_test_2 = [[x] for x in y_test]

            Error_train_nn[k] = np.square(ann.sim(X_train) - y_train_2).sum() / y_train.shape[0]
            Error_test_nn[k] = np.square(ann.sim(X_test) - y_test_2).sum() / y_test.shape[0]

            # figure(k)
            # subplot(1, 2, 1)
            # plot(range(1, len(loss_record)), loss_record[1:])
            # xlabel('Iteration')
            # ylabel('Squared error (crossvalidation)')

            # subplot(1, 3, 3)
            # bmplot(attributeNames, range(1, features_record.shape[1]), -features_record[:, 1:])
            # clim(-1.5, 0)
            # xlabel('Iteration')

        print('Cross validation fold {0}/{1}'.format(k + 1, K))
        # print('Train indices: {0}'.format(train_index))
        # print('Test indices: {0}'.format(test_index))
        print('Features no: {0}\n'.format(selected_features.size))

        k += 1

    print('\n')
    print('Linear regression without feature selection:\n')
    print('- Training error: {0}'.format(Error_train.mean()))
    print('- Test error:     {0}'.format(Error_test.mean()))
    print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum() - Error_train.sum()) / Error_train_nofeatures.sum()))
    print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum() - Error_test.sum()) / Error_test_nofeatures.sum()))
    print('Linear regression with feature selection:\n')
    print('- Training error: {0}'.format(Error_train_fs.mean()))
    print('- Test error:     {0}'.format(Error_test_fs.mean()))
    print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum() - Error_train_fs.sum()) / Error_train_nofeatures.sum()))
    print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum() - Error_test_fs.sum()) / Error_test_nofeatures.sum()))
    print('Neural newtork :\n')
    print('- Training error: {0}'.format(Error_train_nn.mean()))
    print('- Test error:     {0}'.format(Error_test_nn.mean()))
    print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum() - Error_train_nn.sum()) / Error_train_nofeatures.sum()))
    print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum() - Error_test_nn.sum()) / Error_test_nofeatures.sum()))

    figure(k)
    subplot(1, 3, 2)
    bmplot(attributeNamesShorter, range(1, Features.shape[1] + 1), -Features)
    clim(-1.5, 0)
    xlabel('Crossvalidation fold')
    ylabel('Attribute')

    # Inspect selected feature coefficients effect on the entire dataset and
    # plot the fitted model residual error as function of each attribute to
    # inspect for systematic structure in the residual

    f = 2  # cross-validation fold to inspect
    ff = Features[:, f - 1].nonzero()[0]
    if len(ff) is 0:
        print('\nNo features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).')
    else:
        m = lm.LinearRegression(fit_intercept=True).fit(X[:, ff], y)

        y_est = m.predict(X[:, ff])
        residual = y - y_est

        figure(k + 1)
        title('Residual error vs. Attributes for features selected in cross-validation fold {0}'.format(f))
        for i in range(0, len(ff)):
            subplot(2, np.ceil(len(ff) / 2.0), i + 1)
            plot(X[:, ff[i]], residual, '.')
            xlabel(attributeNamesShorter[ff[i]])
            ylabel('residual error')

        show()


if __name__ == '__main__':
    is3D = True
    file = "Cars-file-nice.txt";
    datamatrix = load_from_file(file)
    # create_plots(datamatrix)

    # summary_statistics(datamatrix)

    # if(made1_to_k):
    #    datamatrix_k = convert_using_1_to_k(datamatrix)

    # datamatrix_std, cov, coff = std_cov_coff_matrices(datamatrix_k)
    # create_plots(datamatrix_std, datamatrix_std)
    # svd_graph(datamatrix_std, made1_to_k)
    linear_reg(datamatrix, 0, 10, 10)
    # two_layered_cross_validation(datamatrix, 7, 10, 0)
    # find_best_K(datamatrix, 7)
    # print(find_best_ANN(datamatrix, 7))

    # create_plots(datamatrix, datamatrix_std)
    # svd_graph(datamatrix_std, is3D)

    # CV_BestK.two_layer_cross_validation_k_neighbours(datamatrix, 7, 10, 10)
    # CV_bestnn_classification.two_layer_cross_validation(datamatrix, 7, 5, 5)
    # CV_big_shit.two_layer_cross_validation(datamatrix, 7, 10, 10)
    CV_bestnn_reg.two_layer_cross_validation(datamatrix, 0, 5, 5)
