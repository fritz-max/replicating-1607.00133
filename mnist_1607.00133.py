import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pca import mnist_private_pca_dataloaders, mnist_private_pca

from pyvacy import optim, analysis, sampling

from simpleDPClassifier import Model

noise_levels = [2, 4, 8]

for noise in noise_levels:
    params = {
        'delta': 1e-5,
        'device': 'cpu',
        'iterations': 1000,
        'l2_norm_clip': 4.,
        'l2_penalty': 0.001,
        'lr': 0.052,
        'microbatch_size': 1,
        'minibatch_size': 600,
        'noise_multiplier': noise,
        'pca_components': 60
    }

    X_train, y_train, X_test, y_test = mnist_private_pca(
        epsilon=2, components=params['pca_components'])

    test_loader = torch.utils.data.DataLoader(
        TensorDataset(X_test, y_test), batch_size=64, shuffle=True)

    model = Model(params=params)
    model.train(dataset=TensorDataset(
        X_train, y_train), test_dataset=TensorDataset(X_test, y_test))
    model.eval_dataset(X_test, y_test)
    model.eval_dataset(X_train, y_train)
