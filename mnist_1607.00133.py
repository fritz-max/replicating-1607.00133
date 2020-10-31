"""This script aims to replicate some of the MNIST experiments conducted in
the article Deep Learning with Differential
Privacy(https://arxiv.org/pdf/1607.00133.pdf). Using the pytorch based
library, pyvacy.

The experiments took quite a while to run given that a gradient has to be
computed for each microsample (size=1). Expect it to take approx 4 hours.

Also this does not replicate the experiments fully given the resource
required to do so. (especially the Small noise experiments with around 800
epochs takes a while to run)
"""
from sys import path
path.append("./src")

import numpy as np
import torch
from torch.utils.data import TensorDataset
from pyvacy import optim, sampling

from ppca import mnist_private_pca
from DPClassifier import Model
from differentiallyPrivateSGD import train, create_callback

torch.set_num_threads(1)

PCA_DIM = 60
MINIBATCH_SIZE = 600
MICROBATCH_SIZE = 1
ITERATIONS = 10000
L2_NORM_CLIP=4
LR=0.052
WEIGHT_DECAY=0.001

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
