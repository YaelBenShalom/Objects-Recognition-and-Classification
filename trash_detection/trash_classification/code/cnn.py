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
        output_layer = 6
        self.conv1 = nn.Conv2d(input_layer, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, layer1)
        self.fc2 = nn.Linear(layer1, layer2)
        self.fc3 = nn.Linear(layer2, output_layer)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# class ImageClassificationBase(nn.Module):
#     def training_step(self, batch):
#         images, labels = batch
#         out = self(images)                  # Generate predictions
#         loss = F.cross_entropy(out, labels) # Calculate loss
#         return loss

#     def validation_step(self, batch):
#         images, labels = batch
#         out = self(images)                    # Generate predictions
#         loss = F.cross_entropy(out, labels)   # Calculate loss
#         acc = self.accuracy(out, labels)           # Calculate accuracy
#         return {'val_loss': loss.detach(), 'val_acc': acc}

#     def validation_epoch_end(self, outputs):
#         batch_losses = [x['val_loss'] for x in outputs]
#         epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
#         batch_accs = [x['val_acc'] for x in outputs]
#         epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
#         return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

#     def epoch_end(self, epoch, result):
#         print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
#             epoch+1, result['train_loss'], result['val_loss'], result['val_acc']))

#     def accuracy(self, outputs, labels):
#         _, preds = torch.max(outputs, dim=1)
#         return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        output_layer = 6
        # Use a model resnet50
        self.network = models.resnet50(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, output_layer)

    def forward(self, xb):
        x = torch.sigmoid(self.network(xb))

        return x


# class LitClassifier(pl.LightningModule):
#     def __init__(
#         self, lr: float = 1e-3, num_workers: int = 4, batch_size: int = 32,
#     ):
#         super().__init__()
#         self.lr = lr
#         self.num_workers = num_workers
#         self.batch_size = batch_size

#         self.conv1 = conv_block(3, 16)
#         self.conv2 = conv_block(16, 32)
#         self.conv3 = conv_block(32, 64)

#         self.ln1 = nn.Linear(64 * 26 * 26, 16)
#         self.relu = nn.ReLU()
#         self.batchnorm = nn.BatchNorm1d(16)
#         self.dropout = nn.Dropout2d(0.5)
#         self.ln2 = nn.Linear(16, 5)

#         self.ln4 = nn.Linear(5, 10)
#         self.ln5 = nn.Linear(10, 10)
#         self.ln6 = nn.Linear(10, 5)
#         self.ln7 = nn.Linear(10, 6)

#     def forward(self, img, tab):
#         img = self.conv1(img)
#         img = self.conv2(img)
#         img = self.conv3(img)
#         img = img.reshape(img.shape[0], -1)
#         img = self.ln1(img)
#         img = self.relu(img)
#         img = self.batchnorm(img)
#         img = self.dropout(img)
#         img = self.ln2(img)
#         img = self.relu(img)

#         tab = self.ln4(tab)
#         tab = self.relu(tab)
#         tab = self.ln5(tab)
#         tab = self.relu(tab)
#         tab = self.ln6(tab)
#         tab = self.relu(tab)

#         x = torch.cat((img, tab), dim=1)
#         x = self.relu(x)

#         return self.ln7(x)


# def conv_block(input_size, output_size):
#     block = nn.Sequential(
#         nn.Conv2d(input_size, output_size, (3, 3)), nn.ReLU(), nn.BatchNorm2d(output_size), nn.MaxPool2d((2, 2)),
#     )

#     return block
