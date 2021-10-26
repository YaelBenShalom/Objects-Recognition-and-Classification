import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class BaselineNet(nn.Module):
    """
    This class creates a fully connected neural network for classifying traffic signs
    from the GTSRB dataset.

    Network architecture:
    - Input layer
    - First hidden layer: fully connected layer of size 120 nodes
    - Second hidden layer: fully connected layer of size 84 nodes
    - Output layer: a linear layer with one node per class (in this case 43)

    Activation function: ReLU for both hidden layers
    """

    def __init__(self):
        super(BaselineNet, self).__init__()
        input_layer = 3
        layer1 = 120
        layer2 = 84
        output_layer = 43
        self.conv1 = nn.Conv2d(input_layer, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, layer1)
        self.fc2 = nn.Linear(layer1, layer2)
        self.fc3 = nn.Linear(layer2, output_layer)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        output_layer = 43
        # Use a model resnet50
        self.network = models.resnet50(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, output_layer)

    def forward(self, xb):
        x = torch.sigmoid(self.network(xb))

        return x
