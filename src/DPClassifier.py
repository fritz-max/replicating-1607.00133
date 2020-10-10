import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pca import mnist_private_pca_dataloaders, mnist_private_pca

from pyvacy import optim, analysis, sampling


def _DPSGD(model_params,
           l2_norm_clip=0.001,
           noise_multiplier=2,
           minibatch_size=100,
           microbatch_size=1,
           lr=0.05,
           weight_decay=0.001):
    """Added default values for simplicity
    """
    return optim.DPSGD(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        minibatch_size=minibatch_size,
        microbatch_size=microbatch_size,
        params=model_params,
        lr=lr,
        weight_decay=weight_decay
    )


class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1)
        ).to('cpu')

    def forward(self, X):
        return self.model(X)

    def predict(self, X):
        with torch.no_grad():
            logps = self.model(X)
        return torch.exp(logps)

    def predict_class(self, X):
        return self.predict(X).argmax(dim=1)
