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

#I suggest the ML algoerithm to be K-fold, as it is simple-ish, familiar and the original code also already uses it.

############-------------------------------Remove before submitting
# I took those from the code given - those use pearson and speaman measures. Those are the ones that we need to change. 
############-------------------------------


# ## calculate the pearson and spearman correlations between the word embeddings' similarity scores, using the model's
#    found metric, and the human judgement similarity scores
def get_new_correlations(model_bi, test_loader, cat_name, pearson_data, spearman_data, epoch_cutoff, lowest_loss,
                         duration, big_dataset=False):
    accuracies = []
    ypred = []
    ybatch = []

    # Calculate accuracy
    for X1_batch, X2_batch, y_batch in test_loader:
        y_pred = model_bi(X1_batch, X2_batch)

        ypred.append(y_pred.detach().numpy())
        ybatch.append(y_batch.detach().numpy())

        accuracy = mean_squared_error(y_pred.detach().numpy(), y_batch.detach().numpy())
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




# ## Split the data up into k folds and train k models on that data
def train_k(cat, modelName, n_epochs, pearson_model, spearman_model, path):
    # Print current category and setup necessary variables
    print("Current category is " + cat)
    dataset, train_loader, test_loader = create_dataset(cat)
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
            model_bi, criterion, optimizer = model_setup(dataset)
            model_bi, total_train_losses, model_bool, total_val_losses, epoch_cutoff, lowest_loss_fold = run_epochs(model_bi, criterion, optimizer, n_epochs, train_loader[i], test_loader[i])
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


# ## Initialise the model
def model_setup(dataset):
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
    # model = model.to('cuda') #if you have it

    # set up the loss and the optimizer
    criterion = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model_bi.parameters(), lr=learning_rate)
    return model_bi, criterion, optimizer

#################### Edit when we get the actual dataset    
# ## Create the datasets by splitting the given data up into k folds of data and defining the train and test sets
def create_dataset(cat):
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

