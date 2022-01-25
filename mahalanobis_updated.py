import numpy as np
import torch
from collections import defaultdict
import pandas as pd
import re
from scipy.spatial import distance
import scipy as sp
from matplotlib import pyplot as plt
import scipy.stats


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



def compute_mahala_and_baseline(cat, representations, human_judgements):
    '''
    Function that for each word pair computes the cosine similarity extended with
    the Mahalanobis metric and computes regular cosine similarity.
    Returns (1) a vector containing the Mahalanobis distance between each word pair and
    (2) a vector containing the human judgement similarity score for each word pair.
    '''
    m_sims = [] #similarity between words based on mahalanobis distance
    b_sims = [] #similarity between words based on the baseline, cosine similarity
    h_sims = [] #similarity between words based on human judgements

    cov = np.cov(get_cat_matrix(representations, cat))
    det = np.linalg.det(cov)

    #check whether determinant of the matrix is 0 or not
    if det == 0:
        inv_cov = np.linalg.pinv(cov)
    elif det != 0:
        inv_cov = sp.linalg.inv(cov)

    #get the wordpair
    for wordpair in human_judgements[cat]:

        word1 = wordpair[0][0]
        word2 = wordpair[0][1]

        #find the glove representations for both of these words
        for w1 in representations[cat]:
            for w2 in representations[cat]:
                if w1[0] == word1 and w2[0] == word2:
                    np_w1 = np.array(w1[1])
                    np_w2 = np.array(w2[1])

                    #compute cosine similarity extended with Mahalanobis metric
                    norm_w1 = np.sqrt(np.dot(np.dot(np_w1.T, inv_cov), np_w1))
                    norm_w2 = np.sqrt(np.dot(np.dot(np_w2.T, inv_cov), np_w2))
                    cos_sim_maha = np.dot(np.dot(np_w1.T, inv_cov), np_w2)/(norm_w1 * norm_w2)
                    m_sims.append(cos_sim_maha)

                    #compute cosine similarity as a baseline
                    cos_sim = np.dot(np_w1, np_w2) / (np.linalg.norm(np_w1) * np.linalg.norm(np_w2))
                    b_sims.append(cos_sim)

        #append the human similarity score to the new vector
        h_sims.append(wordpair[1])

    return m_sims, b_sims, h_sims


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
    baseline_sims = defaultdict(list)
    human_sims = defaultdict(list)

    for cat in words:
        # get word similarity based on (1) mahalanobis distance and (2) human judgements represented as vectors
        mahala, baseline, human = compute_mahala_and_baseline(cat, representations, human_judgements)
        mahalanobis_sims[cat] = mahala
        baseline_sims[cat] = baseline
        human_sims[cat] = human

    return words, mahalanobis_sims, baseline_sims, human_sims


def compute_correlations(mahalanobis_sims, baseline_sims, human_sims, words, dims, path):

    all_mahala = []
    all_baseline = []
    all_human = []

    f = open(path + 'Correlation results ' + str(dims) + ' dimensions.txt', 'w')
    f.write('Mahalanobis correlated with human judgements \n')

    for cat in words:
        #make lists of the similarities, ignoring the category they're in
        for i in range(len(mahalanobis_sims[cat])):
            all_mahala.append(mahalanobis_sims[cat][i])
            all_baseline.append(baseline_sims[cat][i])
            all_human.append(human_sims[cat][i])

        #compute correlation per category
        cor_cat = scipy.stats.pearsonr(mahalanobis_sims[cat], human_sims[cat])
        f.write(cat + ': ' + str(cor_cat) + '\n')

    #compute overall correlation
    cor_mahala = scipy.stats.pearsonr(all_mahala, all_human)
    f.write('Overall correlation: ' + str(cor_mahala) + '\n')

    #compute baseline correlations
    f.write('\n')
    f.write('Baseline correlated with human judgements \n')
    for cat in words:
        #correlations per category
        cor_cat_baseline = scipy.stats.pearsonr(baseline_sims[cat], human_sims[cat])
        f.write(cat + ': ' + str(cor_cat_baseline)+ '\n')

    #compute overall baseline correlation
    cor_baseline = scipy.stats.pearsonr(all_baseline, all_human)
    f.write('Overall correlation: ' + str(cor_baseline)+ '\n')

    f.close()


def plot_results(mahalanobis_sims, human_sims, words, dims, path):
    '''
    Function that plots the results of comparing the Mahalanobis
    distances between word pairs with the human judgements.
    '''
    for cat in words:
        plt.scatter(mahalanobis_sims[cat], human_sims[cat], s=5)

    plt.legend(words.keys(), markerscale=2.)
    plt.xlabel('Cosine similarity extended with Mahalanobis metric')
    plt.ylabel('Human judgement similarity')
    plt.savefig(path + 'Scatter_' + str(dims) + 'dims_mahalanobis' )
    plt.close()


def plot_baseline_results(baseline_sims, human_sims, words, dims, path):
    '''
    Function that plots the results of comparing the cosine similarities
    between word pairs with the human judgements.
    '''
    for cat in words:
        plt.scatter(baseline_sims[cat], human_sims[cat], s=5)

    plt.legend(words.keys(), markerscale=2.)
    plt.xlabel('Cosine similarity')
    plt.ylabel('Human judgement similarity')
    plt.savefig(path + 'Scatter_' + str(dims) + 'dims_baseline' )
    plt.close()


dims = 25
glove_reps = 'glove_dim' + str(dims) + '.txt'
#path to where the results should be stored
path = './results/'

words, mahalanobis_distances, baseline_similarities, human_judgements = run_all_categories(glove_reps)
print(mahalanobis_distances)
print(human_judgements)

compute_correlations(mahalanobis_distances, baseline_similarities, human_judgements, words, dims, path)

plot_results(mahalanobis_distances, human_judgements, words, dims, path)
plot_baseline_results(baseline_similarities, human_judgements, words, dims, path)
