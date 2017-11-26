from sklearn.model_selection import train_test_split
from sklearn import cross_validation, tree
import numpy as np
import sklearn.neural_network as nn
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from toolbox_02450 import feature_selector_lr, bmplot
from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim
from sklearn.naive_bayes import MultinomialNB
import sklearn.linear_model as lm
from scipy import stats
from imblearn.over_sampling import RandomOverSampler
from toolbox_02450 import dbplotf
from toolbox_02450 import rocplot, confmatplot
from toolbox_02450 import dbplot, dbprobplot
from bin_classifier_ensemble import BinClassifierEnsemble
from scipy.linalg import svd

attributeNames = ['MPG', 'Cylinders', 'Displacment', 'Horsepower', 'Weight (lbs)', 'Acceleration (MPH)', 'Model year', 'Origin']


def two_layer_cross_validation(input_data, index_to_check, outer_cross_number, inner_cross_number):
    X_outer, y_outer = split_train_test(input_data, index_to_check)
    y_outer = y_outer - 1

    ros = RandomOverSampler(random_state=0)

    N_outer, M_outer = X_outer.shape

    neighbours = 1
    hidden_neurons = 31

    CV_outer = cross_validation.KFold(N_outer, outer_cross_number, shuffle=True)

    test_error_log = list()
    test_error_dt = list()
    test_error_nb = list()
    test_error_k = list()
    test_error_nn = list()
    test_error_mc = list()

    best_log = nn.MLPClassifier(solver='lbfgs', alpha=1e-1, hidden_layer_sizes=(11,), random_state=1)

    bestnn = nn.MLPClassifier(solver='lbfgs', alpha=1e-1, hidden_layer_sizes=(11,), random_state=1)
    bestnn_error = 100000000
    best_model = nn.MLPClassifier(solver='lbfgs', alpha=1e-1, hidden_layer_sizes=(11,), random_state=1)
    best_model_error = 100000000000

    k_outer = 0
    for train_index_outer, test_index_outer in CV_outer:
        X_par = X_outer[train_index_outer, :]
        y_par = y_outer[train_index_outer]

        X_par, y_par = ros.fit_sample(X_par, y_par)

        X_val = X_outer[test_index_outer, :]
        y_val = y_outer[test_index_outer]

        N_inner = len(X_par)

        CV_inner = cross_validation.KFold(len(X_par), inner_cross_number, shuffle=True)

        k = 0
        for train_index_inner, test_index_inner in CV_inner:
            print('Crossvalidation fold: {0}/{1}'.format(k + 1, inner_cross_number))

            X_train = X_par[train_index_inner, :]
            y_train = y_par[train_index_inner]

            X_train, y_train = ros.fit_sample(X_train, y_train)

            X_test = X_par[test_index_inner, :]
            y_test = y_par[test_index_inner]

            log = lm.logistic.LogisticRegression(C=N_inner).fit(X_train, y_train)
            error_2 = 100 * np.sum(y_test.ravel() != log.predict(X_test).ravel()) / y_test.shape[0]
            test_error_log.append(error_2)

            if error_2 < best_model_error:
                best_log = log
                # best_model = log
                best_model_error = error_2

            dt = tree.DecisionTreeClassifier().fit(X_train, y_train)
            error_2 = 100 * np.sum(y_test.ravel() != dt.predict(X_test).ravel()) / y_test.shape[0]
            test_error_dt.append(error_2)

            if error_2 < best_model_error:
                best_model = dt
                best_model_error = error_2

            nb_classifier = MultinomialNB().fit(X_train, y_train)
            y_est_prob = nb_classifier.predict_proba(X_test)
            y_est = np.argmax(y_est_prob, 1)

            error_2 = 100 * np.sum(y_test != y_est) / y_test.shape[0]
            test_error_nb.append(error_2)

            if error_2 < best_model_error:
                # best_model = nb_classifier
                best_model_error = error_2

            clf = nn.MLPClassifier(solver='lbfgs', alpha=1e-1, hidden_layer_sizes=(hidden_neurons,), random_state=1).fit(X_train, y_train)
            error_2 = 100 * np.sum(y_test.ravel() != clf.predict(X_test).ravel()) / y_test.shape[0]
            test_error_nn.append(error_2)

            if error_2 < bestnn_error:
                bestnn = clf
                bestnn_error = error_2

            knclassifier = KNeighborsClassifier(n_neighbors=neighbours).fit(X_train, y_train)
            error_2 = 100 * np.sum(y_test.ravel() != knclassifier.predict(X_test)) / y_test.shape[0]
            test_error_k.append(error_2)

            if error_2 < best_model_error:
                # best_model = knclassifier
                best_model_error = error_2

            counts = np.bincount(y_train.astype(int))
            mostCommon = np.argmax(counts)
            error_2 = 100 * np.sum(y_test.ravel() != mostCommon) / y_test.shape[0]
            test_error_mc.append(error_2)

            k += 1

        # Generalization error
        log = lm.logistic.LogisticRegression(C=N_inner).fit(X_par, y_par)
        y_est = log.predict(X_val)
        test_error_log.append(100 * np.sum(y_est.ravel() != y_val.ravel()) / y_test.shape[0])

        dt = tree.DecisionTreeClassifier().fit(X_par, y_par)
        y_est = dt.predict(X_val)
        test_error_dt.append(100 * np.sum(y_est.ravel() != y_val.ravel()) / y_test.shape[0])

        nb_classifier = MultinomialNB().fit(X_par, np.asarray(y_par))
        y_est_prob = nb_classifier.predict_proba(X_val)
        # y_est = nb_classifier.predict_proba(X_val)
        print('A:', y_est_prob)
        y_est = np.argmax(y_est_prob, 1)
        print('C:', y_est)
        print('D:', np.rint(y_val))
        test_error_nb.append(100 * np.sum(y_est.ravel() != y_val.ravel()) / y_test.shape[0])

        clf = nn.MLPClassifier(solver='lbfgs', alpha=1e-1, hidden_layer_sizes=(hidden_neurons,), random_state=1).fit(X_par, y_par)
        y_est = clf.predict(X_val)
        y_est = np.rint(y_est)
        test_error_nn.append(100 * np.sum(y_est.ravel() != y_val.ravel()) / y_test.shape[0])

        knclassifier = KNeighborsClassifier(n_neighbors=neighbours).fit(X_par, y_par)
        y_est = knclassifier.predict(X_val)
        y_est = np.rint(y_est)
        test_error_k.append(100 * np.sum(y_est.ravel() != y_val.ravel()) / y_test.shape[0])

        counts = np.bincount(y_par.astype(int))
        mostCommon = np.argmax(counts)
        error_2 = 100 * np.sum(y_val.ravel() != mostCommon) / y_val.shape[0]
        test_error_mc.append(error_2)

        print('Test error log: {0}'.format(test_error_log[k_outer]))
        print('Test error dt: {0}'.format(test_error_dt[k_outer]))
        print('Test error nb: {0}'.format(test_error_nb[k_outer]))
        print('Test error k: {0}'.format(test_error_k[k_outer]))
        print('Test error nn: {0}'.format(test_error_nn[k_outer]))
        print('Test error mc: {0}'.format(test_error_mc[k_outer]))
        k_outer += 1

    print('Mean-square error log: {0}'.format(np.mean(test_error_log)))
    print('Mean-square error dt: {0}'.format(np.mean(test_error_dt)))
    print('Mean-square error nb: {0}'.format(np.mean(test_error_nb)))
    print('Mean-square error k: {0}'.format(np.mean(test_error_k)))
    print('Mean-square error nn: {0}'.format(np.mean(test_error_nn)))
    print('Mean-square error mc: {0}'.format(np.mean(test_error_mc)))
    print(best_model)

    out = tree.export_graphviz(best_model, out_file='tree_cars.gvz', feature_names=attributeNames[0:7])

    to_plot_log = [[x] for x in test_error_log]
    to_plot_dt = [[x] for x in test_error_dt]
    to_plot_nb = [[x] for x in test_error_nb]
    to_plot_k = [[x] for x in test_error_k]
    to_plot_nn = [[x] for x in test_error_nn]
    to_plot_mc = [[x] for x in test_error_mc]

    significant_differnece(test_error_dt, test_error_k)

    figure(1)
    plt.boxplot(np.bmat('to_plot_log, to_plot_dt, to_plot_nb, to_plot_k, to_plot_nn, to_plot_mc'))
    xlabel('Log_Reg vs. DT vs. NB vs. K-neighbour vs. NN vs. Most common')
    ylabel('Cross-validation error [%]')

    show()


def significant_differnece(Error_1, Error_2):
    Error_1 = np.asarray(Error_1)
    Error_2 = np.asarray(Error_2)
    K = 10

    z = (Error_1 - Error_2)
    zb = z.mean()
    nu = K - 1
    sig = (z - zb).std() / (K - 1)
    alpha = 0.05

    zL = zb + sig * stats.t.ppf(alpha / 2, nu)
    zH = zb + sig * stats.t.ppf(1 - alpha / 2, nu)

    if zL <= 0 and zH >= 0:
        print('Classifiers are not significantly different')
    else:
        print('Classifiers are significantly different.')


def split_train_test(input_matrix, index):
    y = input_matrix[:, index]
    X = np.delete(input_matrix, index, axis=1)
    print(X.shape)
    return X, y
