import numpy as np
import torch
from collections import defaultdict
import pandas as pd
import re
from scipy.spatial import distance
import scipy as sp
from matplotlib import pyplot as plt


def reduced_glove_reps(words, path):
    '''
    Function that retrieves the reduced GloVe embeddings as made in dim_red.py.
    It retrieves the embeddings that correspond to the given words and organizes
    these embeddings based on the categories they belong to.
    '''
    glove_reps = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            glove_reps[word] = vector

    #create dictionary for categories, words and GloVe representations
    representations = defaultdict(list)

    for category in words:
        for word in words[category].dropna().unique(): #drop NaN values

            word = word.lower() # make words lowercase
            res = torch.from_numpy(glove_reps[word])
            #append word representation
            representations[category].append((word, res))

    return representations



def get_words():
    '''
    Function for retrieving the words (organized in their corresponding categories)
    and the human judgement scores for each possible word pair in each category.
    '''

    words = pd.read_csv('table 1 - word lists.csv')

    human_judgements = defaultdict(list)
    # column headers are word pairs separated by \
    pattern = re.compile('\\\\')

    for category in words:
        if category == "wordsim353":
            break
        # open every file with category name
        filepath = category + "_pairwise.csv"

        # drop "unnamed 0" column
        category_judgements = pd.DataFrame.drop(pd.read_csv(filepath), "Unnamed: 0", axis=1)

        # list of word pairs
        category_pairs = category_judgements.columns

        # for every pair, compute between-subject average similarity
        for pair in category_pairs:
            # average without scaling
            avg_rep = category_judgements.mean(axis=0)[category_judgements.columns.get_loc(pair)]

            # split word pair to a list of two words
            nouns = pattern.split(pair)

            # send human judgements by category and indexed pair of words to tensor
            # strip spaces after splitting
            human_judgements[category].append(([x.strip() for x in nouns], avg_rep/7))
    return words, human_judgements



def compute_mahalanobis(cat, representations, human_judgements):
    '''
    Function that gets the Mahalanobis distance between each word pair in each category.
    Returns (1) a vector containing the Mahalanobis distance between each word pair and
    (2) a vector containing the human judgement similarity score for each word pair.
    '''
    m_sims = [] #similarity between words based on mahalanobis distance
    h_sims = [] #similarity between words based on human judgements

    cov = np.cov(get_cat_matrix(representations, cat))
    #TODO: check if det of matrix is 0
    #inv_cov = sp.linalg.inv(cov)
    inv_cov = np.linalg.pinv(cov)

    #get the wordpair
    for wordpair in human_judgements[cat]:

        word1 = wordpair[0][0]
        word2 = wordpair[0][1]

        #find the glove representations for both of these words
        for w1 in representations[cat]:
            for w2 in representations[cat]:
                if w1[0] == word1 and w2[0] == word2:

                    #compute the mahalanobis distance between the found glove representations and append this to the new vector
                    mahalanobis = sp.spatial.distance.mahalanobis(w1[1], w2[1], inv_cov)
                    m_sims.append(mahalanobis)

        #append the human similarity score to the new vector
        h_sims.append(wordpair[1])

    return m_sims, h_sims


def get_cat_matrix(representations, cat):
    '''
    Function that retrieves the glove embeddings and returns these
    in a matrix that is transposed.
    '''
    matrix = []

    for (_, v) in representations[cat]:
        matrix.append(v.numpy())

    return np.array(matrix).T


def run_all_categories(filename):
    '''
    Function that retrieves the words, glove representations and human
    judgements. Calls compute_mahalanobis to compute the Mahalanobis
    distance between all word pairs and returns this in a dictionary
    along with a dictionary containing the human judgements.
    '''

    # get the words, human judgements and glove representations
    words, human_judgements = get_words()
    representations = reduced_glove_reps(words, filename)

    mahalanobis_sims = defaultdict(list)
    human_sims = defaultdict(list)

    for cat in words:
        # get word similarity based on (1) mahalanobis distance and (2) human judgements represented as vectors
        mahala, human = compute_mahalanobis(cat, representations, human_judgements)
        mahalanobis_sims[cat] = mahala
        human_sims[cat] = human

    return words, mahalanobis_sims, human_sims


def compute_cos_sim(mahalanobis_sims, human_sims, words):
    '''
    Function that computes cosine similarity between mahalanobis
    distance similarities and human judgements per category.
    '''
    cos_sim_per_cat = defaultdict(int)

    for cat in words:
        cos = np.dot(mahalanobis_sims[cat], human_sims[cat])/(np.linalg.norm(mahalanobis_sims[cat])*np.linalg.norm(human_sims[cat]))
        cos_sim_per_cat[cat] = cos

    return cos_sim_per_cat


def plot_results(mahalanobis_sims, human_sims, words):
    '''
    Function that plots the results of comparing the Mahalanobis
    distances between word pairs with the human judgements.
    '''
    #compute the highest Mahalanobis distance to standardize the distances
    max_mahala = max([max(mahalanobis_sims[cat]) for cat in mahalanobis_sims])

    for cat in words:
        #standardize Mahalanobis distances
        normalized_mahala = np.divide(mahalanobis_sims[cat], max_mahala) #TODO: or divide by 7
        #divide each Mahalanobis distance by the corresponding human judgement
        compare = [i / j for i, j in zip(normalized_mahala, human_sims[cat])]

        #plot the results
        plt.plot(sorted(compare))

    plt.legend(words.keys())
    plt.show()


words, mahalanobis_distances, human_judgements = run_all_categories('glove_dim25.txt')
print(mahalanobis_distances)
print(human_judgements)

cos_sim = compute_cos_sim(mahalanobis_distances, human_judgements, words)
print(cos_sim)

plot_results(mahalanobis_distances, human_judgements, words)

