# adding src to path for easy importing
from torch.utils.data import TensorDataset, DataLoader
import torch
from sys import path
path.append("../src")

from simpleDPClassifier import Model
from modelTraining import train

params = {
    'delta': 1e-5,
    'device': 'cpu',
    'iterations': 1,
    'l2_norm_clip': 4.,
    'l2_penalty': 0.001,
    'lr': 0.052,
    'microbatch_size': 1,
    'minibatch_size': 1,
    'noise_multiplier': 2,
    'pca_components': 60
}

X_TRAIN_SHAPE = (60000, 60)
Y_TRAIN_SHAPE = (60000)

X_TEST_SHAPE = (10000, 60)
Y_TEST_SHAPE = (10000)

# generating fake data to test behavior
X_train = torch.zeros(X_TRAIN_SHAPE)
y_train = torch.ones(Y_TRAIN_SHAPE)

X_test = torch.zeros(X_TEST_SHAPE)
y_test = torch.zeros(Y_TEST_SHAPE)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = DataLoader(TensorDataset(X_test, y_test))

def test_Model_creation():
    model = Model(params)


def test_Model_training():
    model = Model(params=params)
    train(model=None,
    mini_batch_loader=None,
    micro_batch_loader=None,
    optimizer=None,
    criterion=None,
    iterations=None,
    device='cpu')
