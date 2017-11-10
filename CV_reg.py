from sklearn.model_selection import train_test_split
from sklearn import cross_validation
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim


def two_layer_cross_validation(input_data, index_to_check, outer_cross_number, inner_cross_number):
    X_outer, y_outer = split_train_test(input_data, index_to_check)

    N_outer, M_outer = X_outer.shape

    CV_outer = cross_validation.KFold(N_outer, outer_cross_number, shuffle=True)

    test_error = list()
    P_plot = list()
    k_outer = 0
    for train_index_outer, test_index_outer in CV_outer:
        X_par = X_outer[train_index_outer, :]
        y_par = y_outer[train_index_outer]
        X_val = X_outer[test_index_outer, :]
        y_val = y_outer[test_index_outer]

        error_matrix = np.zeros(shape=(inner_cross_number, max_neighbours))
        precision_matrix = np.zeros(shape=(inner_cross_number, max_neighbours))

        CV_inner = cross_validation.KFold(len(X_par), inner_cross_number, shuffle=True)

        k = 0
        for train_index_inner, test_index_inner in CV_inner:
            print('Crossvalidation fold: {0}/{1}'.format(k + 1, inner_cross_number))

            X_train = X_par[train_index_inner, :]
            y_train = y_par[train_index_inner]
            X_test = X_par[test_index_inner, :]
            y_test = y_par[test_index_inner]
            size = X_train.shape[0]

            for i in range(max_neighbours):
                knclassifier = KNeighborsClassifier(n_neighbors=i + 1)
                knclassifier.fit(X_train, y_train)
                error_matrix[k][i] = np.square(y_test - knclassifier.predict(X_test)).sum() / y_test.shape[0]
                precision_matrix[k][i] = 100 * np.sum(y_test.ravel() != knclassifier.predict(X_test).ravel()) / y_test.shape[0]
            k += 1

            # Generalization error

        listMean = list()
        for i in range(max_neighbours):
            listMean.append(np.mean(error_matrix[:, i]))
        P_plot.append(listMean)

        index = listMean.index(np.min(listMean))
        print('Optimal amount of neighbours: {0}'.format(index + 1))
        knclassifier = KNeighborsClassifier(n_neighbors=index + 1)
        knclassifier.fit(X_par, y_par)
        y_est = knclassifier.predict(X_val)
        y_est = np.rint(y_est)
        test_error.append(np.square(y_est - y_val).sum().astype(float) / y_test.shape[0])
        print('Test error: {0}'.format(test_error[k_outer]))
        k_outer += 1

    print('Mean-square error: {0}'.format(np.mean(test_error)))

    print_matrix = np.zeros(shape=max_neighbours)
    for l in P_plot:
        print_matrix += l
    print_matrix /= len(P_plot)

    figure()
    plt.plot(range(1, max_neighbours + 1), print_matrix)
    plt.xlabel('Number of neighbours')
    plt.ylabel('Mean Squared Error')
    plt.title('Error rate to neighbours')

    show()


def split_train_test(input_matrix, index):
    y = input_matrix[:, index]
    X = np.delete(input_matrix, index, axis=1)
    print(X.shape)
    return X, y
