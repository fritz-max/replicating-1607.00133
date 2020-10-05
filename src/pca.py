"""
"""

import diffprivlib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms


def mnist_private_pca(epsilon=4):
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    train = datasets.MNIST('data/mnist',
                           train=True,
                           download=True,
                           transform=transformations)

    test = datasets.MNIST('data/mnist',
                          train=False,
                          download=True,
                          transform=transformations)

    X_train = train.data.float().view(-1, 28*28)/255
    y_train = train.targets

    X_test = test.data.float().view(-1, 28*28)/255
    y_test = test.targets

    DPPCA = diffprivlib.models.pca.PCA(
        n_components=60, centered=True, epsilon=epsilon)
    X_train_pc = DPPCA.fit_transform(X_train)
    X_train_pc = torch.Tensor(X_train_pc).view(-1, 60)

    X_test_pc = DPPCA.transform(X_test)
    X_test_pc = torch.Tensor(X_test_pc).view(-1, 60)
    return X_train_pc, y_train, X_test_pc, y_test


def mnist_private_pca_dataloaders(epsilon, batch_size=64, shuffle=True):
    X_train, y_train, X_test, y_test = mnist_private_pca(epsilon=epsilon)
    mnist_train_loader = torch.utils.data.DataLoader(TensorDataset(
        X_train, y_train), batch_size=batch_size, shuffle=shuffle)
    mnist_test_loader = torch.utils.data.DataLoader(TensorDataset(
        X_test, y_test), batch_size=batch_size, shuffle=shuffle)
    return mnist_train_loader, mnist_test_loader
