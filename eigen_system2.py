import numpy as np
import torch

dims = 24
best_fold = 4
cat = "Vehicles"
path = "Categories/Data_per_cat_GLOVE_dim" + str(dims) + "_nc/"
numpypath = path + "weigthsGLOVE_" + cat + "_-4_5_current" + str(best_fold) + ".csv.npy"
data = np.load(numpypath,allow_pickle=True).tolist().weight.detach().numpy()
print(type(data))

metric_matrix = np.dot(data.T,data)
#print(metric_matrix)
eigen_values, eigen_vectors = np.linalg.eig(metric_matrix)

savepath = path + "Eigensystems/"
file = open(savepath + 'eigen_values_' + cat + "_" + str(dims) + '_dimensions.txt', 'w')
np.savetxt(file, eigen_values)
file.close()

file = open(savepath + 'eigen_vectors_' + cat + "_" + str(dims) + '_dimensions.txt', 'w')
np.savetxt(file, eigen_vectors)
file.close()

file = open(savepath + 'eigen_values_max_min_avg_' + cat + "_" + str(dims) + '_dimensions.txt', 'w')
np.savetxt(file, [max(eigen_values), min(eigen_values),sum(eigen_values)/dims])
file.close()
