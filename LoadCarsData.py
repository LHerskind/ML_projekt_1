import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.linalg import svd
#from sklearn import model_selection
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

#from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show

attributeNames = ['MPG','Cylinders','Displacment','Horsepower','Weight (lbs)','Acceleration (MPH)','Model year','Origin']
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
        plt.subplot(2, 4, i+1)
        plt.boxplot(datamatrix[:,i])
        plt.title(attributeNames[i])
        plt.show()

def create_histo(datamatrix):
    plt.figure(figsize=(2, 4))

    for i, color in enumerate(['red', 'yellow', 'blue', 'brown', 'green','cyan','purple', 'orange'], start=0):
        plt.subplot(2, 4, i+1)
        plt.hist(datamatrix[:, i], color=color, edgecolor='black')
        plt.xlabel(attributeNames[i])
        plt.show()

def summary_statistics(datamatrix):
    for i in range(0,8):
        mean_x = datamatrix[:,i].mean()
        std_x = datamatrix[:,i].std(ddof=1)
        median_x = np.median(datamatrix[:,i])
        range_x = datamatrix[:,i].max() - datamatrix[:,i].min()
        max_x = datamatrix[:,i].max()
        min_x = datamatrix[:,i].min()
        twentyfive_x = np.percentile(datamatrix[:,i],25)
        seventyfive_x = np.percentile(datamatrix[:,i],75)
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
    return np.hstack((inputmatrix[:, :7], np.reshape(get_one_to_k_matrix(),(len(datamatrix),3)) ))


def get_one_to_k_matrix():
    return [vectorized(e) for e in datamatrix[:,7]]


def vectorized(j):
    e = np.zeros((3,1))
    e[int(j)-1] = 1.0 / np.sqrt(3)
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
                class_mask = datamatrix_std[:, 7+c].ravel() > 0
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
    U, S, V = svd(datamatrix_std[:,:7], full_matrices=False)
    rho = (S*S)/(S*S).sum()
    rho_cummulative = np.cumsum(rho)

    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(6,4))
        plt.bar(range(1,len(rho)+1), rho, alpha=0.6, align='center', label='Individual explained variance')
        plt.step(range(1,len(rho)+1), rho_cummulative, where='mid', label='Cumulative explained variance')
        plt.ylabel("Explained variance ratio")
        plt.xlabel('Principal components')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

    datamatrix_projected = np.dot(datamatrix_std[:,:7], V.T)

    if(made1_to_k):
        for c in range(0, 3):
            class_mask = datamatrix_std[:, 7 + c].ravel() > 0
            plt.scatter(datamatrix_projected[class_mask, 0], datamatrix_projected[class_mask, 1], label=origins[c])
    else :
        unique_in_matrix = np.unique(datamatrix_std[:, 7])
        for c in range(0, 3):
            class_mask = datamatrix_std[:, 7].ravel() == unique_in_matrix[c]
            plt.scatter(datamatrix_projected[class_mask, 0], datamatrix_projected[class_mask, 1], label=origins[c])

    plt.legend()
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Origins show across PC1 & PC2')
    plt.show()

def split_train_test(input_matrix ,index):
    y = input_matrix[:,index];
    X = np.delete(input_matrix, index, axis=1)
    print (X.shape)
    return X,y

def linear_reg(input_matrix,index, outer_cross_number, inner_cross_number):
    X, y = split_train_test(input_matrix, index)
    N,M = X.shape
    K = outer_cross_number
    #CV = model_selection.KFold(K,True)
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

    k = 0
    for train_index, test_index in CV:
        X_train = X[train_index, :]
        y_train = y[train_index]
        X_test = X[test_index, :]
        y_test = y[test_index]
        internal_cross_validation = inner_cross_number

        Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum()/y_train.shape[0]
        Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum()/y_test.shape[0]

        m = lm.LinearRegression(fit_intercept=True).fit(X_train, y_train)
        Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
        Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]
        textout = '';
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

            figure(k)
            subplot(1, 2, 1)
            plot(range(1, len(loss_record)), loss_record[1:])
            xlabel('Iteration')
            ylabel('Squared error (crossvalidation)')

            subplot(1, 3, 3)
            bmplot(attributeNames, range(1, features_record.shape[1]), -features_record[:, 1:])
            clim(-1.5, 0)
            xlabel('Iteration')

        print('Cross validation fold {0}/{1}'.format(k + 1, K))
        #print('Train indices: {0}'.format(train_index))
        #print('Test indices: {0}'.format(test_index))
        print('Features no: {0}\n'.format(selected_features.size))

        k += 1

    print('\n')
    print('Linear regression without feature selection:\n')
    print('- Training error: {0}'.format(Error_train.mean()))
    print('- Test error:     {0}'.format(Error_test.mean()))
    print('- R^2 train:     {0}'.format(
        (Error_train_nofeatures.sum() - Error_train.sum()) / Error_train_nofeatures.sum()))
    print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum() - Error_test.sum()) / Error_test_nofeatures.sum()))
    print('Linear regression with feature selection:\n')
    print('- Training error: {0}'.format(Error_train_fs.mean()))
    print('- Test error:     {0}'.format(Error_test_fs.mean()))
    print('- R^2 train:     {0}'.format(
        (Error_train_nofeatures.sum() - Error_train_fs.sum()) / Error_train_nofeatures.sum()))
    print(
        '- R^2 test:     {0}'.format((Error_test_nofeatures.sum() - Error_test_fs.sum()) / Error_test_nofeatures.sum()))

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


def find_best_K(input_matrix, index):
    X, y = split_train_test(input_matrix, index)

    L = 40
    ######## We have to roudn otherwise fit won't run #########
    X = np.round(X)
    y = np.round(y)

    N, M = X.shape

    K = 20

    CV = cross_validation.KFold(N,K,shuffle=True)

    #CV = cross_validation.LeaveOneOut(N)
    errors = np.zeros((N, L))
    i = 0
    for train_index, test_index in CV:
        print('Crossvalidation fold: {0}/{1}'.format(i + 1, K))

        # extract training and test set for current CV fold
        X_train = X[train_index, :]
        y_train = y[train_index]
        X_test = X[test_index, :]
        y_test = y[test_index]

        # Fit classifier and classify the test points (consider 1 to 40 neighbors)
        for l in range(1, L + 1):
            knclassifier = KNeighborsClassifier(n_neighbors=l);
            knclassifier.fit(X_train, y_train);
            y_est = knclassifier.predict(X_test);
            errors[i, l - 1] = np.sum(y_est[0] != y_test[0])

        i += 1

    # Plot the classification error rate
    ers = 100 * sum(errors, 0) / N
    best = 0
    bestValue = 100
    for a in range(L):
        if ers[a] < bestValue:
            best = a
            bestValue = ers[a]

 #   print(best+1, bestValue)

    return best+1

def find_best_ANN(input_matrix, index):
    X, y = split_train_test(input_matrix, index)

    L = 20
    ######## We have to roudn otherwise fit won't run #########
    X = np.round(X)
    y = np.round(y)

    N, M = X.shape

    K = 20

    CV = cross_validation.KFold(N,K,shuffle=True)
    errors = np.zeros((N, L))
    i = 0
    for train_index, test_index in CV:
        print('Crossvalidation fold: {0}/{1}'.format(i + 1, K))

        # extract training and test set for current CV fold
        X_train = X[train_index, :]
        y_train = y[train_index]
        X_test = X[test_index, :]
        y_test = y[test_index]

        # Fit classifier and classify the test points (consider 1 to 40 neighbors)
        for l in range(1, L + 1):
            clf = nn.MLPClassifier(solver='lbfgs', alpha=1e-1,
                                   hidden_layer_sizes=(l,), random_state=1)
            clf.fit(X_train, y_train)
            y_est = clf.predict(X_test);
            errors[i, l - 1] = np.sum(y_est[0] != y_test[0])

        i += 1

    # Plot the classification error rate
    ers = 100 * sum(errors, 0) / N
    best = 0
    bestValue = 100
    for a in range(L):
        if ers[a] < bestValue:
            best = a
            bestValue = ers[a]

 #   print(best+1, bestValue)

    return best+1

def two_layered_cross_validation(input_matrix, index, outer_cross_number, inner_cross_number):
    X, y = split_train_test(input_matrix, index)

    ######## We have to roudn otherwise fit won't run #########
    X = np.round(X)
    y = np.round(y)

    N, M = X.shape

    L = find_best_K(input_matrix, index)
    Neurons = find_best_ANN(input_matrix, index)

    K = outer_cross_number
    CV = cross_validation.KFold(N, K, shuffle=True)
    # CV = cross_validation.StratifiedKFold(y.A.ravel(),k=K)

    # Naive Bayes classifier parameters
    alpha = 1.0  # additive parameter (e.g. Laplace correction)
    est_prior = False  # uniform prior (change to True to estimate prior from data)


    # Initialize variables
    Error_logreg = np.empty((K, 1))
    Error_dectree = np.empty((K, 1))
    Error_nb = np.zeros((K,1))
    Error_K = np.zeros((K, 1))
    Error_nn = np.zeros((K,1))
    n_tested = 0

    k = 0
    for train_index, test_index in CV:
        print('CV-fold {0} of {1}'.format(k + 1, K))

        # extract training and test set for current CV fold
        X_train = X[train_index, :]
        y_train = y[train_index]
        X_test = X[test_index, :]
        y_test = y[test_index]


        C = len(np.unique(y_train))

        # Fit and evaluate Logistic Regression classifier
        model = lm.logistic.LogisticRegression(C=N)

        model = model.fit(X_train, y_train)
        y_logreg = model.predict(X_test)
        Error_logreg[k] = 100 * (y_logreg != y_test).sum().astype(float) / len(y_test)

        # Fit and evaluate Decision Tree classifier
        model2 = tree.DecisionTreeClassifier()
        model2 = model2.fit(X_train, y_train)
        y_dectree = model2.predict(X_test)
        Error_dectree[k] = 100 * (y_dectree != y_test).sum().astype(float) / len(y_test)


        #Fit and evaluate naive bayes classifier
        nb_classifier = MultinomialNB(alpha=alpha, fit_prior=est_prior)
        nb_classifier.fit(X_train, y_train)
        y_est_prob = nb_classifier.predict_proba(X_test)
        y_est = np.argmax(y_est_prob, 1)

        Error_nb[k] = 100 * np.sum(y_est != y_test, dtype=float) / y_test.shape[0]

        #Error_nb[k] = 100 * (y_est != y_test).sum().astype(float) / len(y_test)

        # Fit classifier and classify the test points (consider 1 to 40 neighbors)

        knclassifier = KNeighborsClassifier(n_neighbors=L);
        knclassifier.fit(X_train, y_train);
        y_estK = knclassifier.predict(X_test);
        #Error_K[k] = np.sum(y_est[0] != y_test[0])

        Error_K[k] = 100 * (y_estK != y_test).sum().astype(float) / len(y_test)

        clf = nn.MLPClassifier(solver='lbfgs', alpha=1e-1,
                               hidden_layer_sizes=(Neurons,), random_state=1)
        clf.fit(X_train, y_train);
        y_estnn = clf.predict(X_test);
        # Error_K[k] = np.sum(y_est[0] != y_test[0])

        Error_nn[k] = 100 * (y_estnn != y_test).sum().astype(float) / len(y_test)

        k += 1

    # Test if classifiers are significantly different using methods in section 9.3.3
    # by computing credibility interval. Notice this can also be accomplished by computing the p-value using
    # [tstatistic, pvalue] = stats.ttest_ind(Error_logreg,Error_dectree)
    # and test if the p-value is less than alpha=0.05.
    z = (Error_logreg - Error_dectree)
    zb = z.mean()
    nu = K - 1
    sig = (z - zb).std() / (K - 1)
    alpha = 0.05

    zL = zb + sig * stats.t.ppf(alpha / 2, nu);
    zH = zb + sig * stats.t.ppf(1 - alpha / 2, nu);

    if zL <= 0 and zH >= 0:
        print('Classifiers are not significantly different')
    else:
        print('Classifiers are significantly different.')

    # Boxplot to compare classifier error distributions
    figure()
    plt.boxplot(np.bmat('Error_logreg, Error_dectree, Error_nb, Error_K, Error_nn'))
    xlabel('Logistic Regression   vs.   Decision Tree')
    ylabel('Cross-validation error [%]')

    show()

    return 0;

if __name__ == '__main__':
    made1_to_k = True
    file = "Cars-file-nice.txt";
    datamatrix = load_from_file(file)
    # create_plots(datamatrix)

    #summary_statistics(datamatrix)

    if(made1_to_k):
        datamatrix_k = convert_using_1_to_k(datamatrix)

    #datamatrix_std, cov, coff = std_cov_coff_matrices(datamatrix_k)
    #create_plots(datamatrix_std, datamatrix_std)
    #svd_graph(datamatrix_std, made1_to_k)
    #linear_reg(datamatrix, 4, 10, 10)
    two_layered_cross_validation(datamatrix, 7, 10,0)
    #find_best_K(datamatrix, 7)
    #print(find_best_ANN(datamatrix, 7))



