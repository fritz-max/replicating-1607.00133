# adding src to path for easy importing
from sys import path
path.append("../src")

import pca

def test_mnist_private_pca_defaultParamCall():
    pca.mnist_private_pca()

def test_mnist_private_pca_defaultParamCall():
    pca.mnist_private_pca_dataloaders()