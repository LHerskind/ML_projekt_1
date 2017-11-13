from toolbox_02450 import feature_selector_lr, bmplot
import sklearn.linear_model as lm
import neurolab as nl
import numpy as np
from sklearn import cross_validation
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim


def linear_reg(input_matrix, index, outer_cross_number, inner_cross_number):
    X, y = split_train_test(input_matrix, index)
    N, M = X.shape
    K = outer_cross_number
    # CV = model_selection.KFold(K,True)

    neurons = 1
    learning_goal = 25
    max_epochs = 64
    show_error_freq = 65

    CV = cross_validation.KFold(N, K, shuffle=True)

    Features = np.zeros((M, K))
    Error_train = np.empty((K, 1))
    Error_test = np.empty((K, 1))
    Error_train_fs = np.empty((K, 1))
    Error_test_fs = np.empty((K, 1))
    Error_train_mean = np.empty((K, 1))
    Error_test_mean = np.empty((K, 1))
    Error_train_nn = np.empty((K, 1))
    Error_test_nn = np.empty((K, 1))
    k = 0
    for train_index, test_index in CV:
        X_train = X[train_index, :]
        y_train = y[train_index]
        X_test = X[test_index, :]
        y_test = y[test_index]
        internal_cross_validation = inner_cross_number

        Error_train_mean[k] = np.square(y_train - y_train.mean()).sum() / y_train.shape[0]
        Error_test_mean[k] = np.square(y_test - y_test.mean()).sum() / y_test.shape[0]

        m = lm.LinearRegression(fit_intercept=True).fit(X_train, y_train)
        Error_train[k] = np.square(y_train - m.predict(X_train)).sum() / y_train.shape[0]
        Error_test[k] = np.square(y_test - m.predict(X_test)).sum() / y_test.shape[0]
        textout = ''
        selected_features, features_record, loss_record = feature_selector_lr(X_train, y_train, internal_cross_validation, display=textout)

        Features[selected_features, k] = 1
        # .. alternatively you could use module sklearn.feature_selection
        if len(selected_features) is 0:
            print('No features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).')
        else:
            m = lm.LinearRegression(fit_intercept=True).fit(X_train[:, selected_features], y_train)
            Error_train_fs[k] = np.square(y_train - m.predict(X_train[:, selected_features])).sum() / y_train.shape[0]
            Error_test_fs[k] = np.square(y_test - m.predict(X_test[:, selected_features])).sum() / y_test.shape[0]

            y_train_2 = np.asmatrix([[x] for x in y_train])
            y_test_2 = np.asmatrix([[x] for x in y_test])
            ann = nl.net.newff([[-3, 3]] * M, [neurons, 1], [nl.trans.TanSig(), nl.trans.PureLin()])

            ann.train(X_train, y_train_2, goal=learning_goal, epochs=max_epochs, show=show_error_freq)
            y_est_train = ann.sim(X_train)
            y_est_test = ann.sim(X_test)

            Error_train_nn[k] = np.square(y_est_train - y_train_2).sum() / y_train.shape[0]
            Error_test_nn[k] = np.square(y_est_test - y_test_2).sum() / y_test.shape[0]

            figure()
            subplot(2, 1, 1)
            plot(y_train_2, y_est_train, '.')
            subplot(2,1,2)
            plot(y_test_2, y_est_test, '.')
            xlabel('MPG (true, normalized)');
            ylabel('MPG (estimated, normalized)');



        print('Cross validation fold {0}/{1}'.format(k + 1, K))
        print('Features no: {0}\n'.format(selected_features.size))

        k += 1

    print('Feature_select vs. ANN:')
    significant_differnece(Error_1=Error_test_fs, Error_2=Error_test_nn, K=K)
    print('Mean vs. ANN:')
    significant_differnece(Error_1=Error_test_mean, Error_2=Error_test_nn, K=K)
    print('Linear vs. ANN:')
    significant_differnece(Error_1=Error_test, Error_2=Error_test_nn, K=K)

    figure()
    plt.boxplot(np.bmat('Error_test_nn, Error_test_fs, Error_test, Error_train_mean'))
    title('Normalized input/output')
    xlabel('ANN vs. Feature_selected vs. clean vs. mean')
    ylabel('Mean squared error')




    show()


def significant_differnece(Error_1, Error_2, K):
    Error_1 = np.asarray(Error_1)
    Error_2 = np.asarray(Error_2)

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
