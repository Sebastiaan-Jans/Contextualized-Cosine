"""Dimensionality reduction following the recipe of
Raunak, Gupta and Metze (2019).
https://github.com/vyraun/Half-Size/
"""

# import glove_representations
import numpy as np
from sklearn.decomposition import PCA
# from tqdm import tqdm

DEBUG = False

def debugprint(*args):
    if DEBUG:
        print(" ".join(args))

def load_full_glove(embeddings_path):
    """Load the full-dimensionality embedding vectors and return as dict."""

    debugprint("loading GloVe vectors")

    gModel = {}
    with open(embeddings_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            gModel[word] = vector

    debugprint("done")

    return gModel



def reduce_dimension(embeddings, dim, filename, D=0):
    """Reduce dimensionality embeddings in the `embeddings` dict to
    `dim` dimensions. Returns a new dict containing the reduced embeddings.
    Also writes the embeddings to a text file.
    D is a threshold parameter of the post-processing algorithm (PPA).
    Uses the reduction algorithm by Raunak et al. (2019).
    """
    X_train = []
    X_train_names = []
    for word, vector in embeddings.items():
        X_train_names.append(word)
        X_train.append(vector)
    
    X_train = np.asarray(X_train)
    pca_embeddings = {}

    original_dim = X_train.shape[1]
    pca = PCA(n_components=original_dim)

    # the original code simply uses `np.mean(X_train)`,
    # but I'm quite sure it should be the mean vector obtained by
    # averaging along the first axis from the paper
    # this might be because their code is old and uses an old version of numpy
    X_train = X_train - X_train.mean(axis=0)
    X_fit = pca.fit_transform(X_train)
    U1 = pca.components_

    # Remove projections on top components
    z = []
    for i, vector in enumerate(X_train):
        for u in U1[0:D]:
            vector = vector - np.dot(u.transpose(), vector) * u
        z.append(vector)
    
    z = np.array(z)

    # PCA dimensionality reduction to the desired dimensionality
    pca = PCA(n_components=dim)
    # I think here again the mean should be taken over the first axis,
    # but I'm not so sure now
    X_train = z - z.mean(axis=0)
    X_new_final = pca.fit_transform(X_train)

    # PCA for post-processing (PPA) again
    pca = PCA(n_components=dim)
    # again, not sure about the method of averaging
    X_new = X_new_final - X_new_final.mean(axis=0)
    X_new = pca.fit_transform(X_new)
    Ufit = pca.components_

    # again again, not sure about the mean
    X_new_final = X_new_final - X_new_final.mean(axis=0)

    final_pca_embeddings = {}

    with open(filename, 'w', encoding='utf-8') as embedding_file:
        for i, word in enumerate(X_train_names):
            final_pca_embeddings[word] = X_new_final[i]
            embedding_file.write("%s\t" % word)
            for u in Ufit[0:D]:
                final_pca_embeddings[word] = (
                    final_pca_embeddings[word]
                    - np.dot(u.transpose(), final_pca_embeddings[word])
                    * u
                    )
            
            for t in final_pca_embeddings[word]:
                embedding_file.write("%f\t" % t)
            
            embedding_file.write("\n")

    return final_pca_embeddings

def reduce_dimension_flat_means(embeddings, dim, filename, D=0):
    """Reduce dimensionality embeddings in the `embeddings` dict to
    `dim` dimensions. Returns a new dict containing the reduced embeddings.
    Also writes the embeddings to a text file.
    Uses the reduction algorithm by Raunak et al. (2019).
    """
    X_train = []
    X_train_names = []
    for word, vector in embeddings.items():
        X_train_names.append(word)
        X_train.append(vector)
    
    X_train = np.asarray(X_train)
    pca_embeddings = {}

    original_dim = X_train.shape[1]
    pca = PCA(n_components=original_dim)

    # the original code simply uses `np.mean(X_train)`,
    # but I'm quite sure it should be the mean vector obtained by
    # averaging along the first axis from the paper
    # this might be because their code is old and uses an old version of numpy
    X_train = X_train - X_train.mean()
    X_fit = pca.fit_transform(X_train)
    U1 = pca.components_

    # Remove projections on top components
    z = []
    for i, vector in enumerate(X_train):
        for u in U1[0:D]:
            vector = vector - np.dot(u.transpose(), vector) * u
        z.append(vector)
    
    z = np.array(z)

    # PCA dimensionality reduction to the desired dimensionality
    pca = PCA(n_components=dim)
    # I think here again the mean should be taken over the first axis,
    # but I'm not so sure now
    X_train = z - z.mean()
    X_new_final = pca.fit_transform(X_train)

    # PCA for post-processing (PPA) again
    pca = PCA(n_components=dim)
    # again, not sure about the method of averaging
    X_new = X_new_final - X_new_final.mean()
    X_new = pca.fit_transform(X_new)
    Ufit = pca.components_

    # again again, not sure about the mean
    X_new_final = X_new_final - X_new_final.mean()

    final_pca_embeddings = {}

    with open(filename, 'w', encoding='utf-8') as embedding_file:
        for i, word in enumerate(X_train_names):
            final_pca_embeddings[word] = X_new_final[i]
            embedding_file.write("%s\t" % word)
            for u in Ufit[0:D]:
                final_pca_embeddings[word] = (
                    final_pca_embeddings[word]
                    - np.dot(u.transpose(), final_pca_embeddings[word])
                    * u
                    )
            
            for t in final_pca_embeddings[word]:
                embedding_file.write("%f\t" % t)
            
            embedding_file.write("\n")

    return final_pca_embeddings
