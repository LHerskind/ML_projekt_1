import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.linalg import svd
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show

file = "Cars-file-nice.txt";
datamatrix = np.loadtxt(file)
datamatrix[:,0] = datamatrix[:,0] * -1

datamatrix_std = StandardScaler().fit_transform(datamatrix)

cov_matrix = np.cov(datamatrix_std.T)
coff_matrix = np.corrcoef(datamatrix_std.T)

U, S, V = svd(datamatrix_std, full_matrices=False)
rho = (S*S)/(S*S).sum()

figure()
plot(range(1,len(rho)+1),rho, 'o-')

show()


## PCA
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