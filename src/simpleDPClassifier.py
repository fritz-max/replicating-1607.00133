import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pca import mnist_private_pca_dataloaders, mnist_private_pca

from pyvacy import optim, analysis, sampling


class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(params['pca_components'], 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1)
        ).to('cpu')
        self.params = params
        self.optimizer = optim.DPSGD(
            l2_norm_clip=params['l2_norm_clip'],
            noise_multiplier=params['noise_multiplier'],
            minibatch_size=params['minibatch_size'],
            microbatch_size=params['microbatch_size'],
            params=self.model.parameters(),
            lr=params['lr'],
            weight_decay=params['l2_penalty'],
        )
        self.criterion = nn.NLLLoss()
        self.test_results = []

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

    def forward(self, x):
        return self.model(x)

    def train(self, dataset, test_dataset):
        minibatch_loader, microbatch_loader = sampling.get_data_loaders(
            self.params['minibatch_size'],
            self.params['microbatch_size'],
            self.params['iterations']
        )
        print(self.params['noise_multiplier'])

        train_acc = []
        test_acc = []

        iteration = 0
        accum_loss = 0
        for X_minibatch, y_minibatch in minibatch_loader(dataset):
            self.optimizer.zero_grad()
            running_loss = 0
            iteration += 1
            for X_microbatch, y_microbatch in microbatch_loader(TensorDataset(X_minibatch, y_minibatch)):
                X_microbatch = X_microbatch.to(self.params['device'])
                y_microbatch = y_microbatch.to(self.params['device'])

                self.optimizer.zero_microbatch_grad()
                prediction = self.model(X_microbatch)
                loss = self.criterion(prediction, y_microbatch)

                loss.backward()
                self.optimizer.microbatch_step()
                running_loss += loss.item()
                accum_loss += loss.item()
            self.optimizer.step()

            if iteration % 10 == 0:
                train_acc.append(self.eval_dataset(
                    dataset[:][0], dataset[:][1]))
                test_acc.append(self.eval_dataset(
                    test_dataset[:][0], test_dataset[:][1]))
                print('[Iteration %d/%d] [Avg Loss (last 10 It.): %f] [This It. Loss: %f]' %
                      (iteration, self.params['iterations'],
                       accum_loss /
                       (10*(self.params['minibatch_size'] /
                            self.params['microbatch_size'])),
                       running_loss/(self.params['minibatch_size']/self.params['microbatch_size'])))
                accum_loss = 0

        np.save(
            f'accuracy_{self.params["noise_multiplier"]}.npy', np.vstack(
                (np.array(train_acc), np.array(test_acc))))

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
