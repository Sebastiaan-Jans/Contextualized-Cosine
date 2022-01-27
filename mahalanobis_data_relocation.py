import numpy as np
import torch
from collections import defaultdict
import pandas as pd
import re
from scipy.spatial import distance
import scipy as sp
from matplotlib import pyplot as plt
import seaborn as sns


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

    words = pd.read_csv('../table 1 - word lists.csv')

    human_judgements = defaultdict(list)
    # column headers are word pairs separated by \
    pattern = re.compile('\\\\')

    for category in words:
        if category == "wordsim353":
            break
        # open every file with category name
        filepath = "../study1_pairwise_data/data_individual_level/"+category + "_pairwise.csv"

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

def baseline_all_categories(filename):
    
    # get the words, human judgements and glove representations
    words, human_judgements = get_words()
    representations = reduced_glove_reps(words, filename)
    cos_similarity = cosine_sim_cat(words, human_judgements, representations)
    return cos_similarity


def cosine_sim_cat(words, human_judgements, representations):
    '''
    Function that gets the cosine similarithe between each word pair in each category.
    Returns (1) a vector containing the cosine similarity between each word pair for a baseline'''

    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    
    c_similarities = defaultdict(list)
    for cat in words:
            c_sims = [] #similarity between words based on mahalanobis distance
            h_sims = [] #similarity between words based on human judgements
            #get the wordpair
            for wordpair in human_judgements[cat]:

                word1 = wordpair[0][0]
                word2 = wordpair[0][1]

                #find the glove representations for both of these words
                for w1 in representations[cat]:
                    for w2 in representations[cat]:
                        if w1[0] == word1 and w2[0] == word2:
                            cos_sim = cos(w1[1], w2[1])
                            c_sims.append(cos_sim)
            c_similarities[cat] = c_sims
    return c_similarities


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

    plt.ylabel("Ratio mahalanobis distance - human judgements")
    plt.xlabel("Index")
    plt.legend(words.keys())
    #plt.savefig("Ratio mahalanobis distance - human judgements")
    plt.show()

def plot_results_to_baseline(mahalanobis_sims, human_sims, words, path):
    '''Function that plots the mahalanobis distance from human judgement to baseline'''

    #compute the highest Mahalanobis distance to standardize the distances
    max_mahala = max([max(mahalanobis_sims[cat]) for cat in mahalanobis_sims])
    cos_similarity = baseline_all_categories(path)
    for cat in words:
        #standardize Mahalanobis distances
        normalized_mahala = np.divide(mahalanobis_sims[cat], max_mahala) #TODO: or divide by 7
        #divide each Mahalanobis distance by the corresponding human judgement
        compare_mah = [i / j for i, j in zip(normalized_mahala, human_sims[cat])]
        compare_base = [i / j for i, j in zip(cos_similarity[cat], human_sims[cat])]
        differences = [compare_base[i]-compare_mah[i] for i in range(0,len(compare_mah))]
        #db = pd.DataFrame({"base":compare_base,"mah":compare_mah})
        #db = db.sort_values(by="base")
        #plot the results
        #plt.plot(db['base'],db['mah'])
        plt.plot(sorted(differences))
    #plt.xlabel("Baseline distance from human judgement")
    #plt.ylabel("Mahalanobis distance from human judgement")
    plt.ylabel("Difference between Mahalanobis distance and Baseline")
    plt.legend(words.keys())
    plt.savefig("Baseline_minus_Mahalanobis")
    plt.show()


def cosine_sim_cat_matrices(words,path, to_plot = False):
    '''Get the cosine similarity per category to use it as a baseline.'''

    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    representations = reduced_glove_reps(words, path)
    
    all_adjecensy = {}
    for category in words:
        cat_unique = words[category].dropna().unique()
        dim = len(cat_unique)
        adj_matrix = np.zeros((dim, dim))
        for ind1, val1 in enumerate(enumerate(cat_unique)):
            for ind2, val2 in enumerate(enumerate(cat_unique)):
                adj_matrix[ind1, ind2] = cos(representations[category][ind1][1], representations[category][ind2][1])
        all_adjecensy[category] = adj_matrix

        if(to_plot == True):
            print(category)
            ax = sns.heatmap(adj_matrix, linewidth=0.5, vmin=0, vmax=1)
            plt.xticks(range(dim), words[category].dropna().unique(),rotation=-45)
            plt.yticks(range(dim), words[category].dropna().unique(),rotation=45)
            plt.show()

    return all_adjecensy

path = '../glove50-to-25.txt'
words, mahalanobis_distances, human_judgements = run_all_categories(path)#('../glove_dim25.txt')
#print(mahalanobis_distances)
#print(human_judgements)
#cosine_sim_cat(words, path)

#cos_sim_mah = compute_cos_sim(mahalanobis_distances, human_judgements, words)
#print(cos_sim_mah)

#plot_results(mahalanobis_distances, human_judgements, words)
#plot_results_to_baseline(mahalanobis_distances, human_judgements, words, path)
cosine_sim_cat_matrices(words,path, to_plot = False)

