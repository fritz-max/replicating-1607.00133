import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pca import mnist_private_pca_dataloaders, mnist_private_pca

from pyvacy import optim, analysis, sampling


def _DPSGD(model_params):
    """curried DPSGD version, missing model parameters
    """
    return optim.DPSGD(
        l2_norm_clip=0.001,
        noise_multiplier=2,
        minibatch_size=100,
        microbatch_size=1,
        params=model_params,
        lr=0.05,
        weight_decay=0.001
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
        self.criterion = nn.NLLLoss()
        self.test_results = []

    def forward(self, x):
        return self.model(x)

    def eval_dataset(self, X_test, y_test):
        with torch.no_grad():
            logps = self.model(X_test.view(-1, 60))
        return round(((torch.exp(logps).argmax(dim=1) == y_test).sum().float()/len(X_test)*100).item(), 2)

    def eval_dataloader(self, dataloader):
        accuracies = []
        for X_test, y_test in dataloader:
            with torch.no_grad():
                logps = self.model(X_test)
            accuracy = ((torch.exp(logps).argmax(dim=1) ==
                         y_test).sum().float()/len(X_test)*100).item()
            accuracies.append(accuracy)

        test_acc = round(sum(accuracies)/len(accuracies), 2)
        print(
            f"Test Accuracy: {test_acc}%")

        return test_acc





if __name__ == '__main__':
    noise_levels = [2, 4, 8]

    for noise in noise_levels:
        params = {
            'delta': 1e-5,
            'device': 'cpu',
            'iterations': 10000,
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
            X_train, y_train), test_loader=test_loader)
        model.eval_dataset(X_test, y_test)

        with open(f"data_noise{noise}.txt", 'w') as filehandle:
            filehandle.write("Parameters:\n")
            for key, value in params.items():
                filehandle.write(f"{key}: {value}\n")

            filehandle.write("\n")
            filehandle.write(
                f"Noise Level: {noise}\nDP: ({model.dp[0]:.2f}, {model.dp[1]})\n")
            filehandle.write("\n")
            filehandle.write(
                "Test Accuracies (every 100 Iterations, i.e. 1 Epoch):\n")

            for i, elem in enumerate(model.test_results):
                filehandle.write(f"{i}: {elem}\n")

        # self.dp = (analysis.epsilon(
        #                 len(X_train),
        #                 params['minibatch_size'],
        #                 params['noise_multiplier'],
        #                 params['iterations'],
        #                 params['delta']
        #             ),
        #             params['delta'])

        # print(
        #     f"Specified parameters achieve ({self.dp[0]:.2f}, {self.dp[1]})-DP")
