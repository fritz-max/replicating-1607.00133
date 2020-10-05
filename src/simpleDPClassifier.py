import torch
import torch.nn as nn
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

        print("Specified parameters achieve ({}, {})-DP".format(
            analysis.epsilon(
                len(X_train),
                params['minibatch_size'],
                params['noise_multiplier'],
                params['iterations'],
                params['delta']
            ),
            params['delta'],
        ))

    def forward(self, x):
        return self.model(x)

    def train(self, dataset):
        minibatch_loader, microbatch_loader = sampling.get_data_loaders(
            self.params['minibatch_size'],
            self.params['microbatch_size'],
            self.params['iterations']
        )

        iteration = 0
        for X_minibatch, y_minibatch in minibatch_loader(dataset):
            self.optimizer.zero_grad()
            # running_loss = 0
            for X_microbatch, y_microbatch in microbatch_loader(TensorDataset(X_minibatch, y_minibatch)):
                X_microbatch = X_microbatch.to(self.params['device'])
                y_microbatch = y_microbatch.to(self.params['device'])

                self.optimizer.zero_microbatch_grad()
                prediction = self.model(X_microbatch)
                loss = self.criterion(prediction, y_microbatch)

                loss.backward()
                self.optimizer.microbatch_step()
                # running_loss += loss.item()
            self.optimizer.step()

            if iteration % 10 == 0:
                print('[Iteration %d/%d] [Loss: %f]' % (iteration, self.params['iterations'], loss.item()))
            iteration += 1
        
        # for i in range(epochs):
        #     running_loss = 0
        #     for X, y in dataloader:
        #         self.optimizer.zero_grad()
        #         prediction = self.model(X)
        #         loss = self.criterion(prediction, y)
        #         loss.backward()
        #         self.optimizer.step()
        #         running_loss += loss.item()
        #     else:
        #         print("Training loss: ", (running_loss/len(dataloader)))

    def eval_dataloader(self, dataloader):
        accuracies = []
        for X_test, y_test in dataloader:
            with torch.no_grad():
                logps = self.model(X_test)
            accuracy = ((torch.exp(logps).argmax(dim=1) ==
                         y_test).sum().float()/len(X_test)*100).item()
            accuracies.append(accuracy)
        print(
            f"Accuracy: {round(sum(accuracies)/len(accuracies), 2)}%")

    def eval_dataset(self, X_test, y_test):
        with torch.no_grad():
            logps = self.model(X_test.view(-1, params['pca_components']))
        print(f"Accuracy: {round(((torch.exp(logps).argmax(dim=1) == y_test).sum().float()/len(X_test)*100).item(), 2)}%")

if __name__ == '__main__':
    # mnist_train_loader, mnist_test_loader = mnist_private_pca_dataloaders(
    #     epsilon=1)
    params = {
        'delta': 1e-5,
        'device': 'cpu',
        'iterations': 1000,
        'l2_norm_clip': 4.,
        'l2_penalty': 0.001,
        'lr': 0.001,
        'microbatch_size': 1,
        'minibatch_size': 100,
        'noise_multiplier': 2,
        'pca_components': 30
    }

    X_train, y_train, X_test, y_test = mnist_private_pca(epsilon=10, components=params['pca_components'])
    
    trainset = TensorDataset(X_train, y_train)
    # testset = TensorDataset(X_test, y_test)

    model = Model(params=params)
    model.train(dataset=trainset)
    model.eval_dataset(X_test, y_test)
