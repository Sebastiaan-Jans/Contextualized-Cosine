import numpy as np

def load_matrix(path):
    f = open(path, 'r')
    matrix = np.loadtxt(f)
    f.close()

    return matrix

dimensions = [10, 15, 24, 50, 100]
cats = ["Birds","Clothing","Fruit","Furniture","Professions","Sports","Vehicles","Vegetables"]

eigenvalues = {}

for dims in dimensions:
    for cat in cats:
        path = './results/' + str(dims) + 'dims/'
        filename = 'inv_cov_' + cat + '_' + str(dims) + '_dimensions.txt'
        eigen_values, eigen_vectors = np.linalg.eig(load_matrix(path + filename))

        eigenvalues[cat] = [max(eigen_values), min(eigen_values),sum(eigen_values)/dims]

        file = open(path + 'eigen_values_' + cat + '_' + str(dims) + '_dimensions.txt', 'w')
        np.savetxt(file, eigen_values)
        file.close()

        file2 = open(path + 'eigen_vectors_' + cat + '_' + str(dims) + '_dimensions.txt', 'w')
        np.savetxt(file2, eigen_vectors)
        file2.close()

        file3 = open(path + 'eigen_values_max_min_avg_' + cat + '_' + str(dims) + '_dimension.txt', 'w')
        np.savetxt(file3, eigenvalues[cat])
        file3.close()

    #print('Eigen values: ', eigen_values)
    #print('Eigen vectors: ', eigen_vectors[:,0])

        print(dims, cat, eigenvalues[cat])


