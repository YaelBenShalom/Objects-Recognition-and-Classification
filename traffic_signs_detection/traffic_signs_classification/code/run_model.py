import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from read_data import CustomDataLoader


def run_model(model, running_mode='train', train_set=None, valid_set=None, test_set=None,
              batch_size=1, epoch_num=1, learning_rate=0.01, stop_thr=1e-4,
              criterion=nn.CrossEntropyLoss(), device=torch.device('cpu')):
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
      model:                the neural network to be trained or evaluated.
      running_mode:         string, 'train' or 'test'.
      train_set:            the training dataset object generated using the class ReadDataset.
      valid_set:            the validation dataset object generated using the class ReadDataset.
      test_set:             the testing dataset object generated using the class ReadDataset.
      batch_size:           number of training samples fed to the model at each training step.
      learning_rate:        determines the step size in moving towards a local minimum.
      epoch_num:            maximum number of epoch for training the model.
      criterion:            the loss function (runs CrossEntropyLoss() function by default).
      device:               the device the program run on (runs on the CPU by default).
      stop_thr:             if the validation loss from one epoch to the next is less than this value, stop training.
      shuffle:              determines if the shuffle property of the DataLoader is on/off.

    Outputs when running_mode == 'train':
      model:                the trained model.
      train_loss_list:      list of loss value on the entire training dataset.
      valid_loss_list:      list of loss value on the entire validation dataset.
      valid_accuracy_list:  list of accuracy on the entire validation dataset.

    Outputs when running_mode == 'test':
      test_loss:            average loss value on the entire validation or testing dataset.
      test_accuracy:        percentage of correctly classified samples in the validation or testing dataset.

    A summary of the operations this function performs:
    1. Use the DataLoader classes (from read_data.py) to generate training and validation data loaders.
    2. In the training mode:
       - Use the DataLoader class to generate training, validation, or test data loaders.
       - Call the train function for a number of epochs until a stopping criterion is met.
       - Call the test function with the validation data loader at each epoch if the validation
         set is provided.
    3. In the testing mode:
       - Use the DataLoader class to generate test data loader.
       - Call the test function with the test data loader and return the results.
    """

    if running_mode == 'train':
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(
            valid_set, batch_size=batch_size, shuffle=False)

        train_loader = CustomDataLoader(train_loader, to_device)
        valid_loader = CustomDataLoader(valid_loader, to_device)

        return _train(model, train_loader, valid_loader,
                      epoch_num, learning_rate, stop_thr,
                      loss_func=criterion)

    else:
        test_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=False)
        test_loader = CustomDataLoader(test_loader, to_device)

        return _test(model, test_loader, criterion)


def to_device(x, y):
    device = torch.device('cpu')
    x = x.to(device)
    y = y.to(device, dtype=torch.int64)
    return x, y


def loss_batch(model, loss_func, x, y, opt=None):
    loss = loss_func(model(x), y)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(x)


def valid_batch(model, loss_func, x, y):
    output = model(x)
    loss = loss_func(output, y)
    pred = torch.argmax(output, dim=1)
    correct = pred == y.view(*pred.shape)
    return loss.item(), torch.sum(correct).item(), len(x)


def _train(model, train_loader, valid_loader, epoch_num, learning_rate=10e-5, stop_thr=1e-4,
           loss_func=nn.CrossEntropyLoss()):
    """
    This function implements several epochs of training a neural network on a given dataset.

    Inputs:
      model:                the neural network to be trained.
      train_loader:         for loading the network input and targets from the training dataset.
      valid_loader:         for loading the network input and targets from the validation dataset.
      epoch_num:            the number of epoch for training the model
      learning_rate:        determines the step size in moving towards a local minimum
      loss_func:            the loss function (runs CrossEntropyLoss() function by default).
      device:               the device the program run on (runs on the CPU by default).

    Outputs:
      model:                the trained model.
      train_loss_list:      list of loss value on the entire training dataset.
      valid_loss_list:      list of loss value on the entire validation dataset.
      valid_accuracy_list:  list of accuracy on the entire validation dataset.
    """
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    valid_accuracy_list = []
    train_loss_list = []
    valid_loss_list = []
    valid_loss = None

    for epoch in range(epoch_num):
        # Train model
        model.train()
        losses, nums = zip(
            *[loss_batch(model, loss_func, x, y, optimizer) for x, y in train_loader])
        train_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        train_loss_list.append(train_loss)

        # Evaluate model
        model.eval()
        with torch.no_grad():
            losses, corrects, nums = zip(
                *[valid_batch(model, loss_func, x, y) for x, y in valid_loader])
            new_valid_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
            if (valid_loss is not None) and (valid_loss - new_valid_loss < stop_thr):
                break
            valid_loss = new_valid_loss

            valid_accuracy = np.sum(corrects) / np.sum(nums) * 100
            valid_loss_list.append(valid_loss)
            valid_accuracy_list.append(valid_accuracy)

            print(f"Epoch: {epoch + 1}/{epoch_num}\t"
                  f"Train loss: {train_loss:.3f}\t"
                  f"Validation loss: {valid_loss:.3f}\t",
                  f"Validation accuracy: {valid_accuracy:.2f}%")
    return model, train_loss_list, valid_loss_list, valid_accuracy_list


def _test(model, test_loader, loss_func=nn.CrossEntropyLoss()):
    """
    This function evaluates a trained neural network on a validation set or a testing set.

    Inputs:
      model:            trained neural network.
      test_loader:      for loading the network input and targets from the testing dataset.
      loss_func:        the loss function (runs CrossEntropyLoss() function by default).
      device:           the device the program run on (runs on the CPU by default).

    Output:
      test_loss:        average loss value on the entire validation or testing dataset.
      test_accuracy:    percentage of correctly classified samples in the validation or testing dataset.
    """
    model.eval()
    with torch.no_grad():
        losses, corrects, nums = zip(
            *[valid_batch(model, loss_func, x, y) for x, y in test_loader])
        test_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        test_accuracy = np.sum(corrects) / np.sum(nums) * 100

    return test_loss, test_accuracy


def predict(model, features):
    """
    This function evaluates a trained neural network on a validation set or a testing set.

    Inputs:
      model:            trained neural network.
      features:         the features of the testing images .

    Output:
      prediction:       the prediction of the model.
    """
    output = model(features)
    prediction = torch.argmax(output, dim=1)

    return prediction
