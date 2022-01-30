import numpy as np
import torch

dims = 24
path = "Data_per_cat_GLOVE_dim" + str(dims) + "_nc/"
numpypath = path + "weigthsGLOVE_All_-4_5_current1.npy"
data = np.load(numpypath,allow_pickle=True).tolist().weight.detach().numpy()

metric_matrix = np.dot(data.T,data)
#print(metric_matrix)
eigen_values, eigen_vectors = np.linalg.eig(metric_matrix)

file = open(path + 'eigen_values_' + str(dims) + '_dimensions.txt', 'w')
np.savetxt(file, eigen_values)
file.close()

file = open(path + 'eigen_vectors_' + str(dims) + '_dimensions.txt', 'w')
np.savetxt(file, eigen_vectors)
file.close()

file = open(path + 'eigen_values_max_min_avg_' + str(dims) + '_dimensions.txt', 'w')
np.savetxt(file, [max(eigen_values), min(eigen_values),sum(eigen_values)/dims])
file.close()
