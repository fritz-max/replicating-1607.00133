import torch
import torch.nn as nn
import torch.optim as optim


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(60, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1)
        ).to('cpu')
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001)
        self.criterion = nn.NLLLoss()

    def train(self, dataloader, epochs=2):
        for i in range(epochs):
            running_loss = 0
            for X, y in dataloader:
                self.optimizer.zero_grad()
                prediction = self.model(X)
                loss = self.criterion(prediction, y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            else:
                print("Training loss: ", (running_loss/len(dataloader)))

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


if __name__ == '__main__':
    from pca import mnist_private_pca_dataloaders
    mnist_train_loader, mnist_test_loader = mnist_private_pca_dataloaders(
        epsilon=1)
    model = Model()
    model.train(dataloader=mnist_train_loader, epochs=2)
    model.eval_dataloader(mnist_test_loader)
