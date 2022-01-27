"""
Written by: Isa Apallius de Vos, Ghislaine van den Boogerd,
            Mara Fennema and Adriana Correia

Creates and trains a multitude of models based on different vector
representatoins. This code trains per category in the dataset, as opposed  to
training on the entire dataset, as is the case in similarity_whole_dataset.py

"""

import math
import os
import copy
import csv

import re
import scipy
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from collections import defaultdict
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler

from transformers import BertModel, BertForMaskedLM, BertTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import gensim


class similarity_bi(torch.utils.data.Dataset):

    def __init__(self, cat):
        
        # converting the loaded numpy data to torch tensors
        self.X1 = torch.from_numpy(np.concatenate( list(X1.values()), axis=0)).float()
        self.X2 = torch.from_numpy(np.concatenate( list(X2.values()), axis=0)).float()
        self.y = torch.from_numpy(pd.concat( list(data.values()), axis=0).target.to_numpy())

    def __getitem__(self, index):
        # retrieval of a datapoint by index
        X1 = self.X1[index]
        X2 = self.X2[index]
        y = self.y[index].unsqueeze(-1).float()
        return X1, X2, y

    def __len__(self):
        # helper function to check the size of the dataset
        return len(self.y)


# ## Create the datasets by splitting the given data up into k folds of data and defining the train and test sets
def create_dataset(cat, num_folds):
    ## Bilinear

    # get dataset
    dataset = similarity_bi(cat)
    batch_size = 1

    # Setting test size and seed
    test_split = float(1/num_folds)
    shuffle_dataset = True
    random_seed = 42

    # Creating data indices for training and test splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))

    # Shuffling the dataset if need be
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    # set up k-fold
    kf = KFold(n_splits=num_folds, random_state = 42, shuffle = True)
    train_tot = []
    test_tot = []
    for train_split, test_split in kf.split(dataset):
        # Creating the loaders for the train and test data, as defined above
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_split)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_split)
        train_tot.append(train_loader)
        test_tot.append(test_loader)

    return dataset, train_tot, test_tot


# defining the model
class BilinearNN(torch.nn.Module):

    # initializing the model with a certain number of input features
    # output classes, and size of hidden layer(s)
    def __init__(self, n_features, n_classes):
        super().__init__()

        # creating one fully connected layer fc1
        # that applies a linear transformation to the incoming data: y=xA^T +b
        self.fc1 = torch.nn.Bilinear(n_features, n_features, n_classes, bias=False)

    # compute the input values on the fully-connected layer
    def forward(self, inputs1, inputs2):
        return (self.fc1(inputs1,inputs2))/(self.fc1(inputs1,inputs1)*self.fc1(inputs2,inputs2))


# defining our model
class Symmetric_BilinearNN(torch.nn.Module):

    # initializing the model with a certain number of input features, output classes, and size of hidden layer(s)
    def __init__(self, n_features, n_classes):
        super().__init__()

        # creating one fully connected layer fc1
        # that applies a linear transformation to the incoming data: y=xA^T +b
        self.fc1 = torch.nn.Linear(n_features, n_classes, bias=False)

    # x_1^TB^TBx_2/(x_1^TB^TBx_1)*(x_2^TB^TBx_2) = n
    # compute the input values on the fully-connected layer
    def forward(self, inputs1, inputs2):
        return (torch.mm(self.fc1(inputs1),torch.transpose(self.fc1(inputs2),0,1)))/(torch.sqrt(((torch.mm(self.fc1(inputs1),torch.transpose(self.fc1(inputs1),0,1))))*torch.sqrt(torch.mm(self.fc1(inputs2),torch.transpose(self.fc1(inputs2),0,1)))))


# ## Initialise the model
def model_setup(dataset, learning_rate):
    ##Bilinear

    # setting up the model

    # find the number of input features
    n_features = dataset.X1.shape[1] # Length of vector
    n_c = dataset.X1.shape[0] # Number of wordpairs

    # how many hidden units in the network - we'll choose something between the size of the input and output layer
    # see https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
    # for discussion of "best practices"
    hidden_size = 400

    # how many classes in the classification task
    n_classes = n_features
    # for BCELoss you need 1 output unit
    # generally, to find the number of unique labels, use len(set(dataset.y.unique()))

    # instantiate the model
    model_bi = Symmetric_BilinearNN(n_features, n_classes)
    #model_bi = model_bi.to('cuda') #if you have it

    # set up the loss and the optimizer
    criterion = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model_bi.parameters(), lr=learning_rate)
    return model_bi, criterion, optimizer


# ## Train the model for the defined number of epochs
def run_epochs(model_bi, criterion, optimizer, n_epochs, train_loader, val_loader):
    ##Bilinear

    # Initialize variables
    total_train_losses = []
    total_val_losses = []
    bad_model = False
    lowest_epoch_val_loss = 0
    last_epoch_train_loss = []
    last_epoch_train_mse = []
    increasing_loss_counter = 0

    # Run n_epochs times
    for i in range(n_epochs):

        # Train model on X1 and X2 from trainloader
        epoch_train_losses = []
        epoch_train_mse =[]

        for X1_batch, X2_batch, y_batch in train_loader:

            model_bi.train()
            optimizer.zero_grad()

            y_pred = model_bi(X1_batch, X2_batch)

            # In case of a nan, return
            if math.isnan(y_pred.detach().numpy()):
                bad_model = True
                return model_bi, total_train_losses, bad_model, total_val_losses, i+1, epoch_train_loss

            loss = criterion(y_pred, y_batch)

            loss.backward()
            optimizer.step()

            epoch_train_losses.append(loss.item())
            epoch_train_mse.append(mean_squared_error(y_batch.detach().numpy(), y_pred.detach().numpy()))

        # Evaluate model on validation set
        epoch_val_losses = []
        epoch_val_mse =[]

        for X1_batch, X2_batch, y_batch in val_loader:
            model_bi.eval()
            y_pred = model_bi(X1_batch,X2_batch)
            loss = criterion(y_pred, y_batch)

            loss.backward()

            epoch_val_losses.append(loss.item())
            epoch_val_mse.append(mean_squared_error(y_batch.detach().numpy(), y_pred.detach().numpy()))

        # Calculate losses and mse
        epoch_train_loss = np.mean(epoch_train_losses)
        epoch_train_mse = sum(epoch_train_mse)/len(epoch_train_mse)
        epoch_val_loss = np.mean(epoch_val_losses)
        epoch_val_mse = sum(epoch_val_mse)/len(epoch_val_mse)

        # check if the validation loss is increasing
        # if it increases 10 times in a row, stop the training to prevent the model from overfitting
        if epoch_val_loss < 0.1 and lowest_epoch_val_loss <= epoch_val_loss:
            # loss is increasing or the same
            if increasing_loss_counter == 0:
                best_model_bi = copy.deepcopy(model_bi)
            increasing_loss_counter += 1
        elif epoch_val_loss < 0.1 and lowest_epoch_val_loss > epoch_val_loss:
            # loss is decreasing
            increasing_loss_counter = 0
            lowest_epoch_val_loss = epoch_val_loss
        else:
            lowest_epoch_val_loss = epoch_val_loss

        if increasing_loss_counter == 10:
            print(f'Epoch: {i-9}, loss: {epoch_train_loss:.3f}, mse: {epoch_train_mse:.3f}')
            return best_model_bi, total_train_losses, bad_model, total_val_losses, i-9, epoch_train_loss

        # Print the loss every 100th epoch
        if n_epochs != 1 and i % 100 == 0 or i == 999:
            print(f'Epoch: {i+1}, loss: {epoch_train_loss:.3f}, mse: {epoch_train_mse:.3f}')

        total_train_losses.append((sum(epoch_train_losses)/len(epoch_train_losses)))
        total_val_losses.append((sum(epoch_val_losses)/len(epoch_val_losses)))

    return model_bi, total_train_losses, bad_model, total_val_losses, i+1, epoch_train_loss

def save_model_bi_matrix(model_bi,cat, curr_fold, path, lr = -4, fold = 5, model_name = "GLOVE"):
    '''Given a bilinear model, with finsihed traning save the vale matrix'''
    np.save(os.path.join(path, "weigths" + model_name + "_" + cat + "_" + str(lr) + "_" + str(fold) + "_current" + str(curr_fold)), model_bi.fc1)

# ## Split the data up into k folds and train k models on that data
def train_k(cat, modelName, n_epochs, pearson_model, spearman_model, path, num_folds, learning_rate):
    # Print current category and setup necessary variables
    print("Current category is " + cat)
    dataset, train_loader, test_loader = create_dataset(cat, num_folds)
    tot_cor_p = []
    tot_cor_s = []

    # For every fold, run def run_epochs
    # return correlations for each fold and plot loss
    losses = []
    pearsons = []
    spearmans = []
    for i in range(len(train_loader)):
        model_bool = True
        while model_bool is True:
            start = time.time()
            model_bi, criterion, optimizer = model_setup(dataset, learning_rate)
            model_bi, total_train_losses, model_bool, total_val_losses, epoch_cutoff, lowest_loss_fold = run_epochs(model_bi, criterion, optimizer, n_epochs, train_loader[i], test_loader[i])
        save_model_bi_matrix(model_bi, cat, i, path)
        end = time.time()
        dur = end - start
        pearson_model, spearman_model, pearson, spearman = get_new_correlations(model_bi, test_loader[i], cat, pearson_model, spearman_model, epoch_cutoff, lowest_loss_fold, dur)
        plot_losses(cat, modelName, total_train_losses, total_val_losses, i, path)
        losses.append(lowest_loss_fold)
        pearsons.append(pearson[0])
        spearmans.append(spearman[0])
    loss = np.mean(losses)
    pearson = np.mean(pearsons)
    spearman = np.mean(spearmans)
    return pearson_model, spearman_model, loss, pearson, spearman


# ## Plot the training losses and validation losses over the number of epochs that the model needed to train
def plot_losses(cat_name, model_name, train_losses, val_losses, i, path):
    # make a plot with the training losses and validation losses over the epochs
    fig, ax = plt.subplots()
    ax.set_xlabel("Epochs", fontsize=18)
    ax.set_ylabel("Loss", fontsize=18)

    ax.plot(range(len(train_losses)), train_losses, label = cat_name + " train")
    ax.plot(range(len(val_losses)), val_losses, label = cat_name + " val")

    # label the plot
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    # plt.title("Losses over epochs for " + model_name)
    plt.legend()
    plt.savefig(path + "/imgs/" + cat_name + "_Fold_number_{}".format(i+1))
    plt.close()
    return plt


# ## Calculate the pearson and spearman correlations between the word embeddings' similarity scores, using the model's
#    found metric, and the human judgement similarity scores
def get_new_correlations(model_bi, test_loader, cat_name, pearson_data, spearman_data, epoch_cutoff, lowest_loss, duration, big_dataset=False):
    accuracies = []
    ypred=[]
    ybatch=[]

    # Calculate accuracy
    for X1_batch, X2_batch, y_batch in test_loader:
        y_pred = model_bi(X1_batch, X2_batch)

        ypred.append(y_pred.detach().numpy())
        ybatch.append(y_batch.detach().numpy())

        accuracy = mean_squared_error(y_pred.detach().numpy(),y_batch.detach().numpy())
        accuracies.append(accuracy)

    # Get Pearson and Spearman correlations
    ypred = np.concatenate(ypred).ravel()
    ybatch = np.concatenate(ybatch).ravel()
    pearson = scipy.stats.pearsonr(ybatch, ypred)
    spearman = scipy.stats.spearmanr(ybatch, ypred, axis=None)

    # Return calculations in necessary format
    if big_dataset:
        return pearson, spearman
    else:
        print("Model correlations are: ")
        print("Pearson", pearson)
        print("Spearman", spearman)
        p = pearson
        s = spearman
        pearson = [cat_name] + list(pearson) + [epoch_cutoff] + [lowest_loss] + [duration]
        spearman = [cat_name] + list(spearman) + [epoch_cutoff] + [lowest_loss] + [duration]
        pearson_data.append(pearson)
        spearman_data.append(spearman)
        return pearson_data, spearman_data, p, s


# ## Calculate the pearson and spearman correlations between the word embeddings' similarity scores, using the baseline
#    cosine similarity metric, and the human judgement similarity scores
def get_baseline_correlations(test_loader, cat_name, pearson_data, spearman_data):
    pearson_cor = []
    pearson_p = []
    spearman_cor = []
    spearman_p = []

    # predict the cosine similarity for each datapoint in the test set
    for fold in test_loader:
        ypred = []
        ybatch = []
        for X1_batch, X2_batch, y_batch in fold:
            y_pred = torch.mm(X1_batch, torch.transpose(X2_batch, 0, 1)) / (
                        torch.sqrt(torch.mm(X1_batch, torch.transpose(X1_batch, 0, 1))) * torch.sqrt(
                    torch.mm(X2_batch, torch.transpose(X2_batch, 0, 1))))
            ypred.append(y_pred.detach().numpy())
            ybatch.append(y_batch.detach().numpy())

        # Get Pearson and Spearman correlations
        ypred = np.concatenate(ypred).ravel()
        ybatch = np.concatenate(ybatch).ravel()
        pearson_cor.append(scipy.stats.pearsonr(ybatch, ypred)[0])
        pearson_p.append(scipy.stats.pearsonr(ybatch, ypred)[1])
        spearman_cor.append(scipy.stats.spearmanr(ybatch, ypred, axis=None)[0])
        spearman_p.append(scipy.stats.spearmanr(ybatch, ypred, axis=None)[1])

    print("Baseline correlations for cat ", cat_name)
    print("Pearson", np.mean(pearson_cor), "\t p", np.mean(pearson_p))
    print("Spearman", np.mean(spearman_cor), "\t p", np.mean(spearman_p))

    # save and return the correlations
    pearson = [cat_name] + [np.mean(pearson_cor)] + [np.mean(pearson_p)]
    spearman = [cat_name] + [np.mean(spearman_cor)] + [np.mean(spearman_p)]
    pearson_data.append(pearson)
    spearman_data.append(spearman)
    return pearson_data, spearman_data


# ## General Word Embedding Functions
#print word pair similarities for each category

def cosineSim(words):
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    for category in words:
        cat_unique = words[category].dropna().unique()
        dim = len(cat_unique)
        adj_matrix = np.zeros((dim, dim))
        for ind1, val1 in enumerate(enumerate(cat_unique)):
            for ind2, val2 in enumerate(enumerate(cat_unique)):
                adj_matrix[ind1, ind2] = cos(representations[category][ind1][1], representations[category][ind2][1])

        print(category)

        ax = sns.heatmap(adj_matrix, linewidth=0.5, vmin=0, vmax=1)
        plt.xticks(range(dim), words[category].dropna().unique(),rotation=-45)
        plt.yticks(range(dim), words[category].dropna().unique(),rotation=45)
        # plt.show()


#function that compares embedding model and human of two nouns of a category
def compare(category, n1, n2, ctxt=False):
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    if ctxt:
        repr = representations_ctxt
    else:
        repr = representations
    df = words[category]

    #get index of words in function call:
    ind1 = df.index[df==n1][0]
    ind2 = df.index[df==n2][0]

    #get human judgements from finding the nouns on the list:
    for i in range(len(human_judgements[category])):
        if (human_judgements[category][i][0][0] == n1 and human_judgements[category][i][0][1] == n2) or (human_judgements[category][i][0][0] == n2 and human_judgements[category][i][0][1] == n1):

            #return ratio between cosine similarity and human judgement
            return (cos(repr[category][ind1][1], repr[category][ind2][1])/human_judgements[category][i][1]).item()

    return None


#store relative differences in dictionary per category
def relative(category, ctxt=False):
    relative = {}
    for nouns, val in human_judgements[category]:
        relative.update({nouns[0] + " " + nouns[1]: compare(category, nouns[0], nouns[1], ctxt)} )
    return relative

def passMaskedSentence(sent, modelLM, tokenizer):
    # returns embedding of word + context
    input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokenizer.tokenize(sent.replace('mask', '[MASK]')) + ['[SEP]'])
    tens = torch.LongTensor(input_ids).unsqueeze(0)
    return modelLM(tens)[0][0]

def predictMaskedWord(sent, modelLM, tokenizer):
    # predicts masked word in a sentence
    result = passMaskedSentence(sent, modelLM, tokenizer)
    ordered = torch.sort(result[sent.split().index("mask") + 1])
    return list(map(lambda i: tokenizer.convert_ids_to_tokens(ordered[1][i].item()), range(768)))


# ## retrieve the human judgement scores and the wordsim and simlex data (only if non-contextualised embeddings are used)
def get_words(model_name, context_on):
    '''
    Function for retrieving the words (organized in their corresponding categories)
    and the human judgement scores for each possible word pair in each category.
    '''

    words = pd.read_csv('../table 1 - word lists.csv')
    #-----------------below comment starts here
    human_judgements = defaultdict(list)
    # column headers are word pairs separated by \
    pattern = re.compile('\\\\')

    for category in words:
        if category == "wordsim353":
            break
        # open every file with category name
        filepath = "../study1_pairwise_data/data_individual_level/" + category + "_pairwise.csv"

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
    '''    # tensor to encode human judgements
    if not context_on:
        # add WordSimilarity-353 dataset to human judgement data
        category = "wordsim353"

        # open every file with category name
        with open("WordSimilarity-353.csv") as f:
            reader = csv.reader(f)
            judgements = list(reader)

        for i in range(1, len(judgements)):
            human_judgements[category].append(([judgements[i][0], judgements[i][1]], float(judgements[i][2])/10))

        # add SimLex-999 dataset to human judgement data
        category = "simlex999"

        # open every file with category name
        with open("SimLex-999.csv") as f:
            reader = csv.reader(f)
            judgements = list(reader)

        for i in range(1, len(judgements)):
            human_judgements[category].append(([judgements[i][0], judgements[i][1]], float(judgements[i][2])/10))
    '''
    return False, words, human_judgements


# ## retrieve the embeddings for the selected embedding type
def get_model(words, dims=200, model_name = "GLOVE", context_on=False):
    # Ensures that context_on can only be True when model is GPT-2 or BERT
    if (model_name != "GPT-2" and model_name != "BERT") and context_on:
        context_on = False

    if model_name == "GLOVE":
        print("We only use GLOVE")
        return glove_model(words, dims)
    elif model_name == "word2vec":
        return word2vec_model()
    elif model_name == "GPT-2":
        return gpt2_model(context_on)
    elif model_name == "BERT":
        return bert_model(context_on)
    else:
        raise CustomError("Trying to use a model that is not known. Please choose between GLOVE, word2vec, GPT-2 or BERT.")


# ## GloVe Code
def glove_model(words, dims):
    #load the pretrained GloVe model (this may take some time)
    gModel = {}
    
    with open('../dim_reduces_embeddings/glove_dim' + str(dims) + '.txt', 'r', encoding='utf-8') as f:
    #with open("glove.6B.200d.txt", 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            gModel[word] = vector


    # ## Getting GloVe pretrained representations

    #create list for categories, words and GloVe representations
    representations = defaultdict(list)
    
    for category in words:

        for word in words[category].dropna().unique(): #drop NaN values
            word = word.lower() # make words lowercase
            #save vector from pretrained word2vec model
            res = torch.from_numpy(gModel[word])

            #append word representation
            representations[category].append((word, res))

    return representations


# ## Word2Vec Code
def word2vec_model():
    #load the pretrained Word2Vec model (this may take some time)
    wModel = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

    ## Getting Word2Vec pretrained representations
    #create list for categories, words and Word2Vec representations
    representations = defaultdict(list)

    for category in words:

        for word in words[category].dropna().unique(): #drop NaN values

            #save vector from pretrained word2vec model
            res = torch.from_numpy(wModel[word])

            #append word representation
            representations[category].append((word, res))
    return representations


# ## GPT-2 Code
def predict_gpt2(sent, modelLM, tokenizer):
    # Predict the next word based on given sentence sent using GPT-2

    # Tokenize sentence
    indexed_tokens = tokenizer.encode(sent)
    tokens_tensor = torch.tensor([indexed_tokens])
    modelLM.eval()

    # Get all possible predictions
    with torch.no_grad():
        outputs = modelLM(tokens_tensor)
        predictions = outputs[0]

    # Get the 5 most likely prediction indices
    predicted_indices = torch.topk(predictions[0, -1, :], k=5)
    pred_ind = predicted_indices[1]

    # Get indexes from given tensors and add to list
    indices_list = [x.item() for x in pred_ind]

    # Get words at given indexes
    contexts = [tokenizer.decode(index) for index in indices_list]

    return contexts

def gpt2_model(context=False):
    mName = 'gpt2-medium'
    gpt2Model = GPT2LMHeadModel.from_pretrained(mName)
    tokenizer = GPT2Tokenizer.from_pretrained(mName)
    if context:
        return gpt2_model_ctxt(gpt2Model, tokenizer)
    else:
        return gpt2_model_no_ctxt(gpt2Model, tokenizer)

def gpt2_model_no_ctxt(gpt2Model, tokenizer):
    ## Getting GPT-2 pretrained representations
        #create list for categories, words and GPT-2 representations
    representations = defaultdict(list)

    for category in words:

        for word in words[category].dropna().unique(): #drop NaN values

            #tokenize and convert to indices
            text_index = tokenizer.encode(word, add_prefix_space=True)
            res = gpt2Model.transformer.wte.weight[text_index,:]
            rep = sum(res)/len(res)
            #append word representation
            representations[category].append((word, rep))
    return representations


# Due to computational constraints, contextualised GPT-2 cannot be run all at once, but only works one category at a time. Please set a category below.
# Possible categories are:
# Birds, Clothing, Fruit, Furniture, Professions, Sports, Vegetables, Vehicles
def gpt2_model_ctxt(gpt2Model, tokenizer, gpt2_category="Birds"):
    if gpt2_category not in ['Birds', 'Clothing', 'Fruit', 'Furniture', 'Professions', 'Sports', 'Vegetables', 'Vehicles']:
        raise CustomError("You did not use one of the eight possible categories for the gpt2 category. Please use one of the following: 'Birds', 'Clothing', 'Fruit', 'Furniture', 'Professions', 'Sports', 'Vegetables', 'Vehicles'.")
    # give example predictions for all categories
    for category in words:
        print(category)

        if category in ['Furniture', 'Clothing', 'Fruit']:
            sent = "People think the " + category + " is very"
        else:
            sent = "People think the " + category + " are very"

        contexts=predict_gpt2(sent, gpt2Model, tokenizer)
        print(contexts)

    # Get context for the categories using GPT-2

    category = gpt2_category
    #create list for categories, words and Bert representations
    representations_ctxt = defaultdict(list)

    print(category)

    # ensure correct grammar based on singular or plural category name
    if category in ['Furniture', 'Clothing', 'Fruit']:
        sent = "People think the " + category + " is very"
    else:
        sent = "People think the " + category + " are very"

    contexts = predict_gpt2(sent, gpt2Model, tokenizer)
    tens = torch.LongTensor(tokenizer.encode(sent)).unsqueeze(0)
    res = gpt2Model(tens)[0][0]

    for word in words[category].dropna().unique(): #drop NaN values
        rep_ctxt = torch.zeros(len(res[0]))
        for context in contexts:
            #tokenize and convert to indices
            sent = context + word
            tens = torch.LongTensor(tokenizer.encode(sent)).unsqueeze(0)

            #retrieve pre-trained representations
            res = gpt2Model(tens)[0][0]

            #tensor for word representation
            index = len(tens[0].numpy())
            rep = torch.zeros(len(res[0]))

            #these lines are still from the BERT contextualized code and need to be updated for GPT-2.
            #averages over all representations of possible sub-word tokens (all tokens between '[CLS]' and 'is a category [SEP]' (5 tokens))
            for i in range(index-1):
                rep += res[i+1]

            avg_rep = rep/(index-1)
            rep_ctxt += avg_rep

        #append word representation
        representations_ctxt[category].append((word, rep_ctxt/len(contexts)))
    return representations_ctxt


# ## BERT Code
def bert_model(context=False):
    mName = 'bert-base-uncased'
    bModel = BertModel.from_pretrained(mName)
    tokenizer = BertTokenizer.from_pretrained(mName, do_lower_case=True)
    if context:
        return bert_model_ctxt(bModel, tokenizer)
    else:
        return bert_model_no_ctxt(bModel, tokenizer)

def bert_model_no_ctxt(bModel, tokenizer):
    #Change model name depending on which Bert version you want to use (bert-large-uncased or bert-base-uncased)
    ## Getting Bert pretrained representations

    #create list for categories, words and Bert representations
    representations = defaultdict(list)

    for category in words:

        for word in words[category].dropna().unique(): #drop NaN values

            #tokenize and convert to indices
            tens = torch.LongTensor(tokenizer.encode(word)).unsqueeze(0)
            #retrieve pre-trained representations
            res = bModel(tens)[0][0]
            #tensor for word representation
            index = len(tens[0].numpy())
            rep = torch.zeros(len(res[0]))

            #averages over all representations of possible sub-word tokens (all tokens between '[CLS]' and '[SEP]')
            for i in range(index-2):
                rep += res[i+1]

            #append word representation
            representations[category].append((word, rep/(index-2)))
    return representations

# ## Getting Contextualized Bert pretrained representations
def bert_model_ctxt(bModel, tokenizer):
    # Get context for the categories using BERT

    #create list for categories, words and Bert representations
    representations_ctxt = defaultdict(list)

    for category in words:
        if category == "wordsim353":
            break

        sent = "mask "  +  category

        contexts = predictMaskedWord(sent, bModel, tokenizer)[-5:]

        tens = torch.LongTensor(tokenizer.encode(sent)).unsqueeze(0)
        res = bModel(tens)[0][0]

        for word in words[category].dropna().unique(): #drop NaN values
            rep_ctxt = torch.zeros(len(res[0]))
            for context in contexts:
                #tokenize and convert to indices
                sent = context + word
                tens = torch.LongTensor(tokenizer.encode(sent)).unsqueeze(0)

                #retrieve pre-trained representations
                res = bModel(tens)[0][0]

                #tensor for word representation
                index = len(tens[0].numpy())
                rep = torch.zeros(len(res[0]))

                #averages over all representations of possible sub-word tokens (all tokens between '[CLS]' and 'is a category [SEP]' (5 tokens))
                for i in range(index-3):
                    rep += res[i+2]

                avg_rep = rep/(index-3)
                rep_ctxt += avg_rep

            #append word representation
            representations_ctxt[category].append((word, rep_ctxt/len(contexts)))
    return representations_ctxt


# ## Prepare data for ML
def prepare_data(words, representations, human_judgements, model_name, context_on):
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    data = defaultdict(list)
    X = defaultdict(list)
    X1 = defaultdict(list)
    X2 = defaultdict(list)

    # Loop through each category
    for category in words:
        if category == "wordsim353" and context_on:
            break
        if model_name == "GPT-2" and context_on:
            category = gpt2_category
        hm_len = len(human_judgements[category])
        rep_ctxt_len = len(representations[category][0][1])


        X[category] = np.zeros((hm_len, rep_ctxt_len))
        X1[category] = np.zeros((hm_len, rep_ctxt_len))
        X2[category] = np.zeros((hm_len, rep_ctxt_len))

        df = words[category]

        i = 0
        for nouns, val in human_judgements[category]:
            #get index of words in function call:
            # print(representations[category][df.index].shape)
            rep1 = representations[category][df.index[df==nouns[0]][0]][1]
            rep2 = representations[category][df.index[df==nouns[1]][0]][1]

            #cosine similarity between the representations of those words:
            point = rep1 * rep2

            norm = torch.norm(rep1) * torch.norm(rep2)

            data[category].append([nouns[0] + " " + nouns[1], cos(rep1, rep2).item(), torch.norm(point).item(), val])
            X[category][i] = (point/norm).detach().numpy()
            X1[category][i] = rep1.detach().numpy()
            X2[category][i] = rep2.detach().numpy()
            i += 1

        data[category] = pd.DataFrame(data[category])
        data[category].columns=["n1_n2", "Cosine_Sim", "Inner_Product", "target"]

        if model_name == "GPT-2" and context_on:
            break

    return data, X, X1, X2


# ## Save the model and baseline results to four separate csv files
def save_data(path, cat, model_name, pearson_model, spearman_model, pearson_base, spearman_base, lr, fold):

    # Model results with pearson correlations
    with open(os.path.join(path, "pearson_MODEL_" + model_name + "_" + cat + "_" + str(lr) + "_" + str(fold) + ".csv"), "w", newline = '') as f:
        writer = csv.writer(f)
        writer.writerows(pearson_model)

    # Model results with spearman correlations
    with open(os.path.join(path, "spearman_MODEL_" + model_name + "_" + cat + "_" + str(lr) + "_" + str(fold) + ".csv"), "w", newline = '') as f:
        writer = csv.writer(f)
        writer.writerows(spearman_model)

    # Baseline results with pearson correlations
    with open(os.path.join(path, "pearson_BASE_" + model_name + "_" + cat + "_" + str(lr) + "_" + str(fold) + ".csv"), "w", newline = '') as f:
        writer = csv.writer(f)
        writer.writerows(pearson_base)
    # Baseline results with spearman correlations
    with open(os.path.join(path, "spearman_BASE_" + model_name + "_" + cat + "_" + str(lr) + "_" + str(fold) + ".csv"), "w", newline = '') as f:
        writer = csv.writer(f)
        writer.writerows(spearman_base)


# ## Create a directory to save the data in
def make_save_dir(model_name, dim = 200, context_on = False):
    if context_on and (model_name == "GPT-2" or model_name == "BERT"):
        ctxt_str = "_contextualised"
    elif model_name == "GPT-2" or model_name == "BERT":
        ctxt_str = "_nc"
    else:
        ctxt_str = "_nc"
    path = os.path.join(os.getcwd(), "Data_per_cat_" + model_name + "_dim" + str(dim) + ctxt_str)
    if not os.path.isdir(path):
        os.mkdir(path)
    if not os.path.isdir(path + "/imgs/"):
        os.mkdir(path + "/imgs/")
    return path, ctxt_str


# ## Train and test a model on the selected category
def run_category(cat, model_name, n_epochs, pearson_model, spearman_model, pearson_base, spearman_base, path, num_folds,learning_rate, ctxt_str):
    pearson_model, spearman_model, loss, pearson, spearman = train_k(cat, model_name, n_epochs, pearson_model, spearman_model, path, num_folds, learning_rate)
    dataset, train_loader, test_loader = create_dataset(cat, num_folds)
    pearson_base, spearman_base = get_baseline_correlations(test_loader, cat, pearson_base, spearman_base)

    # save best lr+fold combo in dicts
    model_cat = model_name + '-' + ctxt_str + ' ' + cat
    lr_fold = str(learning_rate) + ' ' + str(num_folds)
    loss_dict[model_cat][lr_fold] = loss
    corr_p_dict[model_cat][lr_fold] = pearson
    corr_s_dict[model_cat][lr_fold] = spearman
    return pearson_model, spearman_model, pearson_base, spearman_base


# ## Train a separate model on each of the categories in the category list
def run_all_categories(words, model_name, n_epochs, pearson_model, spearman_model, pearson_base, spearman_base, path, num_folds, learning_rate, ctxt_str):
    for cat in words:
        pearson_model, spearman_model, pearson_base, spearman_base = run_category(cat, model_name, n_epochs, pearson_model, spearman_model, pearson_base, spearman_base, path, num_folds, learning_rate, ctxt_str)
    cat = "all_categories"
    save_data(path, cat, model_name, pearson_model, spearman_model, pearson_base, spearman_base, learning_rate, num_folds)


"""
RUN PER CATEGORY HERE
"""

# define which embeddings you want to use for training
# the first variable is a string with the name of the embedding type (options: GPT-2, word2vec, BERT and GLOVE)
# the second variable is a boolean that indicates whether you want to use the contextualised embeddings for that
# embedding type (currently only possible for BERT)
all_models = [("GPT-2", False), ("word2vec", False), ("BERT", False), ("GLOVE", False), ("BERT", True)]

# define dictionaries to save the hyperparameter results
loss_dict = defaultdict(defaultdict)
corr_p_dict = defaultdict(defaultdict)
corr_s_dict = defaultdict(defaultdict)



# small function to get the key of a dict given a value 
def get_key(val, dict):
    for key, value in dict.items():
         if val == value:
             return key

def find_best_params():
    labels_best = ["model name", "category", "lowest loss lr", "lowest loss folds", "highest pearson lr", "highest pearson folds", "highest spearman lr", "highest spearman folds"]
    best_params = []
    best_params.append(labels_best)


    # for-loop to find the best hyperparameter combination per model and category
    # Dictionaries are used to save all the different model/cat/lr/fold combinations, this for-loop loops through them
    # to find the lr+fold combo with the lowest loss and highest correlations for every model and category combination
    for key in loss_dict.keys():
        model, cat = key.split()
        lowest_loss = float('inf')
        highest_corr_p = float('-inf')
        highest_corr_s = float('-inf')
        for value in loss_dict[key].values():
            if value < lowest_loss:
                lowest_loss = value
        lr_loss, folds_loss = get_key(lowest_loss, loss_dict[key]).split()
        for value in corr_p_dict[key].values():
            if value > highest_corr_p:
                highest_corr_p = value
        lr_corr_p, folds_corr_p = get_key(highest_corr_p, corr_p_dict[key]).split()
        for value in corr_s_dict[key].values():
            if value > highest_corr_s:
                highest_corr_s = value
        lr_corr_s, folds_corr_s = get_key(highest_corr_s, corr_s_dict[key]).split()
        best = [model] + [cat]+ [lr_loss] + [folds_loss] + [lr_corr_p] + [folds_corr_p] + [lr_corr_s] + [folds_corr_s]
        best_params.append(best)
    print(best_params)


    # save the best hyperparameter combinations in a csv file
    with open("hyperparameter_results_per_cat.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(best_params)


def hyperp_glove():
    model_name = "GLOVE"
    reduction_option = [50] #[15, 24,50]
    ### Since it takes forever, 2 hours for one condition on the 15 dim, and it will only get worse time wise, I will not experiment with num_folds and learning_rate.
    ### Besides, they look pretty random and it does not look like it is the best idea to customize per category. So, I am just going to use the forst ones. 
    for num_folds in [5]:#[5, 6, 7]:
        for learning_rate in [0.00001]:#[0.00001, 0.000001, 0.0000001]:
            print("\nlr =", learning_rate, "and num_folds =", num_folds)
            for dim in reduction_option:
                # initialise the pearson and spearman lists for each model
                labels_model = ["category", "correlation", "p-value", "epochs", "lowest train loss", "duration"]
                labels_base = ["category", "correlation", "p-value"]
                pearson_model = []
                spearman_model = []
                pearson_model.append(labels_model)
                spearman_model.append(labels_model)
                pearson_base = []
                spearman_base = []
                pearson_base.append(labels_base)
                spearman_base.append(labels_base)

                # get the human judgements and representations and prepare the data
                context_on, words, human_judgements = get_words(model_name, False)
                representations = get_model(words, dim, model_name, context_on)
                path, ctxt_str = make_save_dir(model_name, dim = dim)
                global data
                global X
                global X1
                global X2
                data, X, X1, X2 = prepare_data(words, representations, human_judgements, model_name, context_on)

                # define the maximal number of epochs for training
                max_epochs = 500

                # define the category types
                #categories = ['Birds', 'Clothing', 'Fruit', 'Furniture', 'Professions', 'Sports', 'Vegetables', 'Vehicles']
                categories = ['All']
                # for each category in the list of categories, train a model on the words in that category
                run_all_categories(categories, model_name, max_epochs, pearson_model, spearman_model, pearson_base, spearman_base, path, num_folds, learning_rate, ctxt_str)


hyperp_glove()
find_best_params()