# adding src to path for easy importing
from sys import path
path.append("../src")

import torch
from pyvacy import optim, analysis, sampling
from torch.utils.data import TensorDataset, DataLoader
from DPClassifier import Model, _DPSGD
from modelTraining import train

delta = 1e-5
device = 'cpu'
l2_norm_clip = 4.
l2_penalty = 0.001
lr = 0.052
pca_components = 60
minibatch_size = 600
microbatch_size = 1
iterations = 1
noise_multiplier = 2

X_TRAIN_SHAPE = (60000, 60)
Y_TRAIN_SHAPE = (60000)

X_TEST_SHAPE = (10000, 60)
Y_TEST_SHAPE = (10000)

INPUT_DIM = 60

# generating fake data to test behavior
X_train = torch.zeros(X_TRAIN_SHAPE)
y_train = torch.ones(Y_TRAIN_SHAPE)

X_test = torch.zeros(X_TEST_SHAPE)
y_test = torch.zeros(Y_TEST_SHAPE)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

minibatch_loader_func, microbatch_loader_func = sampling.get_data_loaders(
    minibatch_size, microbatch_size, iterations
)

minibatch_loader = minibatch_loader_func(train_dataset)

def test_Model_creation():
    model = Model(input_dim=INPUT_DIM)

def test_Model_training():
    model = Model(input_dim=INPUT_DIM)
    train(
        model=model,
        minibatch_loader=minibatch_loader,
        micro_loader_func=microbatch_loader_func,
        optimizer=_DPSGD(model.parameters()),
        criterion=torch.nn.NLLLoss(),
        iterations=1,
        device='cpu')

def test_Model_prediction():
    model = Model(input_dim=INPUT_DIM)
    train(
        model=model,
        minibatch_loader=minibatch_loader,
        micro_loader_func=microbatch_loader_func,
        optimizer=_DPSGD(model.parameters()),
        criterion=torch.nn.NLLLoss(),
        iterations=1,
        device='cpu')
    model.predict(X_test)