import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import Tensor


def run_model(model, running_mode='train', train_set=None, valid_set=None, test_set=None,
              batch_size=1, learning_rate=0.01, epoch_num=1, criterion=nn.CrossEntropyLoss(),
              stop_thr=1e-4, shuffle=True):
    """
    This function either trains or evaluates a model.

    training mode:  the model is trained and evaluated on a validation set, if provided.
                    If no validation set is provided, the training is performed for a fixed number of epochs.
                    Otherwise, the model should be evaluated on the validation set at the end of each epoch and the
                    training should be stopped based on one of these two conditions (whichever happens first):
                    1. The validation loss stops improving.
                    2. The maximum number of epochs is reached.

    testing mode:   the trained model is evaluated on the testing set.

    Inputs:

    model:          the neural network to be trained or evaluated.
    running_mode:   string, 'train' or 'test'.
    train_set:      the training dataset object generated using the class ReadDataset.
    valid_set:      the validation dataset object generated using the class ReadDataset.
    test_set:       the testing dataset object generated using the class ReadDataset.
    batch_size:     number of training samples fed to the model at each training step.
    learning_rate:  determines the step size in moving towards a local minimum.
    epoch_num:       maximum number of epoch for training the model.
    stop_thr:       if the validation loss from one epoch to the next is less than this value, stop training.
    shuffle:        determines if the shuffle property of the DataLoader is on/off.

    Outputs when running_mode == 'train':

    model:          the trained model.
    loss:           dictionary with keys 'train' and 'valid'.
                    The value of each key is a list of loss values. Each loss value is the average of
                    training/validation loss over one epoch.
                    If the validation set is not provided just return an empty list.
    acc:            dictionary with keys 'train' and 'valid'.
                    The value of each key is a list of accuracies (percentage of correctly classified samples in the
                    dataset). Each accuracy value is the average of training/validation accuracies over one epoch.
                    If the validation set is not provided just return an empty list.

    Outputs when running_mode == 'test':

    loss:           the average loss value over the testing set.
    accuracy:       percentage of correctly classified samples in the testing set.

    A summary of the operations this function performs:
    1. Use the DataLoader class to generate training, validation, or test data loaders.
    2. In the training mode:
       - define an optimizer (we use SGD in this homework).
       - call the train function (see below) for a number of epochs until a stopping criterion is met.
       - call the test function (see below) with the validation data loader at each epoch if the validation set is
         provided.

    3. In the testing mode:
       - call the test function (see below) with the test data loader and return the results.
    """

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    device = torch.device('cpu')
    if running_mode == 'train':
        valid_loss = None
        train_loss_list, valid_loss_list, train_accuracy_list, valid_accuracy_list = [], [], [], []
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)

        if valid_set is not None:
            valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=shuffle)

        for epoch in range(epoch_num):
            model, train_loss, train_accuracy = _train(model, train_loader, optimizer, criterion, device=device)
            train_loss_list.append(train_loss)
            train_accuracy_list.append(train_accuracy)

            if valid_set is not None:
                new_valid_loss, valid_accuracy = _test(model, valid_loader, criterion, device=device)
                valid_loss_list.append(new_valid_loss)
                valid_accuracy_list.append(valid_accuracy)
                if (valid_loss is not None) and (valid_loss - new_valid_loss < stop_thr):
                    break
                valid_loss = new_valid_loss
            else:
                valid_loss_list, valid_accuracy_list = [], []

        loss = {
            'train': train_loss_list,
            'valid': valid_loss_list
        }
        accuracy = {
            'train': train_accuracy_list,
            'valid': valid_accuracy_list
        }

        return model, loss, accuracy

    else:
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)
        test_loss, test_accuracy = _test(model, test_loader, criterion, device=device)

        return test_loss, test_accuracy


def _train(model, data_loader, optimizer, criterion, device=torch.device('cpu')):
    """
    This function implements ONE EPOCH of training a neural network on a given dataset.
    I used nn.CrossEntropyLoss() for the loss function.

    Inputs:
    model:          the neural network to be trained.
    data_loader:    for loading the network input and targets from the training dataset.
    optimizer:      the optimization method, e.g., SGD.
    device:         I run everything on the CPU.

    Outputs:
    model:          the trained model.
    train_loss:     average loss value on the entire training dataset.
    train_accuracy: average accuracy on the entire training dataset.
    """

    running_loss = 0.0
    correct = 0.0
    total = 0.0

    for i, (inputs, targets) in enumerate(data_loader):
        # clear the gradients
        optimizer.zero_grad()
        # compute the model output
        yhat = model(inputs.float())
        # calculate loss
        loss = criterion(yhat, targets)
        # credit assignment
        loss.backward()
        # update model weights
        optimizer.step()

        running_loss += loss.item()

        yhat1 = yhat.detach().numpy()
        # convert to class labels
        yhat_labels = np.argmax(yhat1, axis=1)
        # Add to the correct number of predictions
        correct += len(yhat_labels) - np.count_nonzero(yhat_labels - targets.numpy())
        total += len(targets)

    train_loss = running_loss / len(data_loader)
    train_accuracy =  correct / total * 100

    return model, train_loss, train_accuracy


def _test(model, data_loader, criterion, device=torch.device('cpu')):
    """
    This function evaluates a trained neural network on a validation set or a testing set.
    I used nn.CrossEntropyLoss() for the loss function.

    Inputs:
    model:          trained neural network.
    data_loader:    for loading the network input and targets from the validation or testing dataset.
    device:         I run everything on CPU.

    Output:
    test_loss:      average loss value on the entire validation or testing dataset.
    test_accuracy:  percentage of correctly classified samples in the validation or testing dataset.
    """

    running_loss = 0.0
    correct = 0.0
    total = 0.0

    for i, (inputs, targets) in enumerate(data_loader):
        # clear the gradients
        # optimizer.zero_grad()
        # compute the model output
        yhat = model(inputs.float())
        # calculate loss
        loss = criterion(yhat, targets)
        # credit assignment
        # loss.backward()
        # update model weights
        # optimizer.step()
        running_loss += loss.item()

        yhat1 = yhat.detach().numpy()
        # convert to class labels
        yhat_labels = np.argmax(yhat1, axis=1)
        # Add to the correct number of predictions
        correct += len(yhat_labels) - np.count_nonzero(yhat_labels - targets.numpy())
        total += len(targets)

    test_loss = running_loss / len(data_loader)
    test_accuracy = correct / total * 100

    return test_loss, test_accuracy
