from collections import defaultdict
import pandas as pd
import torch
import numpy as np

def glove_model():
    """Load the pretrained GloVe embedding vectors. Path is hardcoded.
    Returns a defaultdict with as keys the word categories, and as values
    lists of tuples which contain a word string as the first element and
    the corresponding embedding as the second element.
    """
    gModel = {}
    with open("glove.6B.200d.txt", 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            gModel[word] = vector

    # ## Getting GloVe pretrained representations

    # create list for categories, words and GloVe representations
    representations = defaultdict(list)

    for category in words:

        for word in words[category].dropna().unique():  # drop NaN values
            word = word.lower()  # make words lowercase
            # save vector from pretrained word2vec model
            res = torch.from_numpy(gModel[word])

            # append word representation
            representations[category].append((word, res))

    return representations



words = pd.read_csv('table 1 - word lists.csv')

if __name__ == "__main__":
    glove_reps = glove_model()
    #print(glove_reps)
    #print(glove_reps['Furniture'][0][1])
