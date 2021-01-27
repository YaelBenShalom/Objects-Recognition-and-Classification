import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):
    """
    This class creates a fully connected neural network for classifying traffic signs
    from the GTSRB dataset.

    Network architecture:
    - Input layer
    - First hidden layer: fully connected layer of size 128 nodes
    - Second hidden layer: fully connected layer of size 64 nodes
    - Output layer: a linear layer with one node per class (in this case 10)

    Activation function: ReLU for both hidden layers
    """

    def __init__(self):
        super(Net, self).__init__()
        size_input_layer = 3
        size_layer1 = 120
        size_layer2 = 84
        size_output_layer = 43
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, size_layer1)
        self.fc2 = nn.Linear(size_layer1, size_layer2)
        self.fc3 = nn.Linear(size_layer2, size_output_layer)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()