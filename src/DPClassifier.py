import torch

class Model(torch.nn.Module):
    '''Simple dense model, with one hidden layer
    '''
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
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
