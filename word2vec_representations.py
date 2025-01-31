import gensim
from collections import defaultdict
import pandas as pd
import torch

def word2vec_model():
    # load the pretrained Word2Vec model (this may take some time)
    wModel = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)

    ## Getting Word2Vec pretrained representations
    # create list for categories, words and Word2Vec representations
    representations = defaultdict(list)

    for category in words:

        for word in words[category].dropna().unique():  # drop NaN values

            # save vector from pretrained word2vec model
            res = torch.from_numpy(wModel[word])

            # append word representation
            representations[category].append((word, res))
    return representations

def dim_reduction_word2vec(representations, desired_dim):
    for category in words:
        for word in range(len(words[category].dropna().unique())):
            representations[category][word][1] #last 1 is to get the tensor representation and not the word itself
            #TODO: reduce this 1x300 tensor to 25 dimensions


words = pd.read_csv('table 1 - word lists.csv')
word2vec_reps = word2vec_model()
#print(word2vec_reps)
#print(word2vec_reps['Furniture'][0][1])