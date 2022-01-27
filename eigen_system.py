import numpy as np

def load_matrix(path):
    f = open(path, 'r')
    matrix = np.loadtxt(f)
    f.close()

    return matrix

dimensions = [10, 15, 24, 25, 50, 100]

for dims in dimensions:
    path = './results/' + str(dims) + 'dims/'
    filename = 'inv_cov_' + str(dims) + '_dimensions.txt'
    eigen_values, eigen_vectors = np.linalg.eig(load_matrix(path + filename))

    file = open(path + 'eigen_values' + str(dims) + '_dimensions.txt', 'w')
    np.savetxt(file, eigen_values)
    file.close()

    file2 = open(path + 'eigen_vectors' + str(dims) + '_dimensions.txt', 'w')
    np.savetxt(file2, eigen_vectors)
    file2.close()

print('Eigen values: ', eigen_values)
print('Eigen vectors: ', eigen_vectors[:,0])
