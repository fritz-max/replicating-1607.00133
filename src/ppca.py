"""This file suppies two methods for extracting a private Principle Component
Analysis of the mnist dataset.

Notes:
For this the IBM diffprivlib is used, which is
used to greatly reduced the dimensionality of the mnist dataset and decrease
the time experiments take to run.
"""

import diffprivlib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms


def mnist_private_pca(epsilon=4, components=60, download=True):
    '''function that returns the principal components of the features of the
    MNIST dataset in addition ot the labels.
    '''
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    train = datasets.MNIST('data/mnist',
                           train=True,
                           download=download,
                           transform=transformations)

    test = datasets.MNIST('data/mnist',
                          train=False,
                          download=download,
                          transform=transformations)

    # flatten X from 2d [28,28] to 1d [784]
    X_train = train.data.view(-1, 28*28)
    y_train = train.targets
    X_test = test.data.view(-1, 28*28)
    y_test = test.targets

    # fit and transform X_train
    DPPCA = diffprivlib.models.pca.PCA(
        n_components=components, centered=True, epsilon=epsilon, data_norm=True)
    X_train_pc = DPPCA.fit_transform(X_train)
    X_train_pc = torch.Tensor(X_train_pc).view(-1, components)

    # transform X_test using the same linear combinations of the feature as
    # learned from X_train
    X_test_pc = DPPCA.transform(X_test)
    X_test_pc = torch.Tensor(X_test_pc).view(-1, components)
    return X_train_pc, y_train, X_test_pc, y_test


def mnist_private_pca_dataloaders(epsilon=4, components=60, batch_size=64, shuffle=True, download=True):
    '''function that returns the principal components of the features of the
    MNIST dataset in addition ot the labels in pytorch dataloaders.
    '''
    # wrapping mnist_private_pca as dataloaders and returning them
    X_train, y_train, X_test, y_test = mnist_private_pca(
        epsilon=epsilon, components=components, download=download)
    mnist_train_loader = torch.utils.data.DataLoader(TensorDataset(
        X_train, y_train), batch_size=batch_size, shuffle=shuffle)
    mnist_test_loader = torch.utils.data.DataLoader(TensorDataset(
        X_test, y_test), batch_size=batch_size, shuffle=shuffle)
    return mnist_train_loader, mnist_test_loader
