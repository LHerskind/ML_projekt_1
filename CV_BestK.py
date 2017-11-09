from sklearn.model_selection import train_test_split
from sklearn import cross_validation


def two_layer_cross_validation_k_neighbours(input_data, index_to_check, outer_cross_number, inner_cross_number):
    X_outer, y_outer = split_train_test(input_data, index_to_check)

    max_neighbours = 40

    N_outer, M_outer = X_outer.shape

    CV_outer = cross_validation.KFold(N_outer, outer_cross_number, shuffle=True)

    test_error = list()
    k_outer = 0
    for train_index_outer, test_index_outer in CV_outer:
        X_par = X_outer[train_index_outer, :]
        y_par = y_outer[train_index_outer]
        X_val = X_outer[test_index_outer, :]
        y_val = y_outer[test_index_outer]

        error_matrix = np.zeros(shape=(inner_cross_number, max_neighbours))

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
            k += 1

            # Generalization error
        Error_gen = list()

        for i in range(max_neighbours):
            sum = 0.0
            for j in range(inner_cross_number):
                sum += (float(size) / X_par.shape[0]) * error_matrix[j][i]
            Error_gen.append(sum)
        print(Error_gen)

        index = Error_gen.index(np.min(Error_gen))
        plot()
        print('Optimal amount of hidden units: {0}'.format(index + 1))
        knclassifier = KNeighborsClassifier(n_neighbors=index + 1)
        knclassifier.fit(X_par, y_par)
        y_est = knclassifier.predict(X_val)
        y_est = np.rint(y_est)

        test_error.append(np.power(y_est - y_val, 2).sum().astype(float) / y_test.shape[0])
        print('Test error: {0}'.format(test_error[k_outer]))
        k_outer += 1

    print('Mean-square error: {0}'.format(np.mean(test_error)))



def split_train_test(input_matrix, index):
    y = input_matrix[:, index];
    X = np.delete(input_matrix, index, axis=1)
    print(X.shape)
    return X, y
