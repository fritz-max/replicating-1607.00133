from sys import path
path.append("./src")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pca import mnist_private_pca

from pyvacy import optim, sampling

from DPClassifier import Model


PCA_DIM = 60
MINIBATCH_SIZE = 600
MICROBATCH_SIZE = 1
ITERATIONS = 10000
L2_NORM_CLIP=4
LR=0.052
WEIGHT_DECAY=0.001

def train(model, minibatch_loader, micro_loader_func, optimizer, criterion, callback, callback_per_iteration, device='cpu'):
    """TODO::
    """
    # sample a Lot

    iteration = 0
    for X_minibatch, y_minibatch in minibatch_loader:
        running_loss = 0 
        # the DPSGD optimizer uses two gradients
        # setting the first to zero
        optimizer.zero_grad()

        for X_microbatch, y_microbatch in micro_loader_func(TensorDataset(X_minibatch, y_minibatch)):

            # moving data onto device
            X_microbatch = X_microbatch.to(device)
            y_microbatch = y_microbatch.to(device)

            # zeroing second gradient
            optimizer.zero_microbatch_grad()

            prediction = model.forward(X_microbatch)
            loss = criterion(prediction, y_microbatch)

            loss.backward()
            optimizer.microbatch_step()
            running_loss += loss.item()

        optimizer.step()
        # print(f"Loss: {running_loss/len(minibatch_loader)}")

        if iteration % callback_per_iteration == 0:
            print("Iterations: ", iteration)
            callback()     
        
        iteration += 1


def get_accuracy(predictions, targets):
    return round((sum((predictions == targets)).float()/len(predictions)*100).item(), 2)

def create_callback(model, train_dataset, test_dataset, train_acc_list, test_acc_list):
    
    def eval_callback():
        # print(train_dataset[:][0].shape)
        train_predictions = model.predict_class(train_dataset[:][0])
        train_accuracy = get_accuracy(train_predictions, train_dataset[:][1])
        train_acc_list.append(train_accuracy)

        test_predictions = model.predict_class(test_dataset[:][0])
        test_accuracy = get_accuracy(test_predictions, test_dataset[:][1])
        test_acc_list.append(test_accuracy)

        print(f"Training Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")

    return eval_callback

for NOISE_MULTIPLIER in [2,4,8]:
    X_train, y_train, X_test, y_test = mnist_private_pca(
        epsilon=2, components=PCA_DIM)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    minibatch_loader_func, microbatch_loader_func = sampling.get_data_loaders(
        MINIBATCH_SIZE, MICROBATCH_SIZE, ITERATIONS
    )

    minibatch_loader = minibatch_loader_func(train_dataset)

    train_accuracies, test_accuracies = [], []

    model = Model(input_dim=PCA_DIM)

    optimizer = optim.DPSGD(
            l2_norm_clip=L2_NORM_CLIP,
            noise_multiplier=NOISE_MULTIPLIER,
            minibatch_size=MINIBATCH_SIZE,
            microbatch_size=MICROBATCH_SIZE,
            params=model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY
        )

    train(
        model=model,
        minibatch_loader=minibatch_loader,
        micro_loader_func=microbatch_loader_func,
        optimizer=optimizer,
        criterion=torch.nn.NLLLoss(),
        callback=create_callback(model, train_dataset, test_dataset, train_accuracies, test_accuracies),
        callback_per_iteration=10,
        device='cpu')

    np.save(f"./results/train_accuracies_noise{NOISE_MULTIPLIER}", train_accuracies)
    np.save(f"./results/test_accuracies_noise{NOISE_MULTIPLIER}", test_accuracies)

