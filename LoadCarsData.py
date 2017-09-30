import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.linalg import svd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
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
        plt.ylabel(attributeNames[i])
    plt.show()


def create_histo(datamatrix):
    plt.figure(figsize=(2, 4))

    for i, color in enumerate(['red', 'yellow', 'blue', 'brown', 'green','cyan','purple', 'orange'], start=0):
        plt.subplot(2, 4, i+1)
        plt.hist(datamatrix[:, i], color=color, edgecolor='black')
        plt.xlabel(attributeNames[i])

    plt.show()


def convert_using_1_to_k(inputmatrix):
    return np.hstack((inputmatrix[:, :7], np.reshape(get_one_to_k_matrix(),(len(datamatrix),3)) ))


def get_one_to_k_matrix():
    return [vectorized(e) for e in datamatrix[:,7]]


def vectorized(j):
    e = np.zeros((3, 1))
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
    U, S, V = svd(datamatrix_std, full_matrices=False)
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

    datamatrix_projected = np.dot(datamatrix_std, V.T)

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


if __name__ == '__main__':
    made1_to_k = True
    file = "Cars-file-nice.txt";
    datamatrix = load_from_file(file)

    if (made1_to_k):
        datamatrix_k = convert_using_1_to_k(datamatrix)

    datamatrix_std, cov, coff = std_cov_coff_matrices(datamatrix_k)

    create_plots(datamatrix, datamatrix_std)
    #create_plots(datamatrix_std, datamatrix_std)

    svd_graph(datamatrix_std, made1_to_k)


'''  ## PCA
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
eig_pairs.sort(key=lambda x: x[0], reverse=True)

PCAs = 4

print(eig_pairs[0][1].shape)

#matrix_w = eig_pairs[0][1].reshape(8, 1)
for i in range(1, PCAs):
    matrix_w = np.hstack((matrix_w, eig_pairs[i][1].reshape(8, 1)))

#for line in file_object.readlines():
#   print (line)
'''
