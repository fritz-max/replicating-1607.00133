"""This file defines a simple model similar (one hidden layer of 1000
units) to the one used in the paper.

Notes:
The reason why this dense architecture was reasonable to use on the
mnist dataset, is due to the fact that the data has been reduced to a 60-dim
vector using a private PCA. This was done to greatly increase the speed at
which the experiments can be conducted.
"""
import torch

class Model(torch.nn.Module):
    '''Simple dense model, with one hidden layer
    '''
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 10),
            torch.nn.LogSoftmax(dim=1)
        ).to('cpu')

    def forward(self, X):
        '''Used to propagate the features through the model to later update
        model by backpropagation
        '''
        return self.model(X)

    def predict(self, X):
        '''Use model to predict labels given that input features X. Returns
        the probabilities of each sample being each class.
        '''
        with torch.no_grad():
            logps = self.model(X)
        return torch.exp(logps)

    def predict_class(self, X):
        '''Use model to predict labels given that input features X. Returns
        predicted class.
        '''
        return self.predict(X).argmax(dim=1)
