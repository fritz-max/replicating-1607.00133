import torch

class Model(torch.nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
            torch.nn.LogSoftmax(dim=1)
        ).to('cpu')

    def forward(self, X):
        return self.model(X)

    def predict(self, X):
        with torch.no_grad():
            logps = self.model(X)
        return torch.exp(logps)

    def predict_class(self, X):
        return self.predict(X).argmax(dim=1)
