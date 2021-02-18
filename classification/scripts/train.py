import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt


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


def fit(model, train_loader, valid_loader, epoch_num, learning_rate=10e-5,
        loss_func=nn.CrossEntropyLoss(), device=torch.device('cpu')):
    """
    This function trains a model.
    """
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    train_accuracy_list = []
    valid_accuracy_list = []
    train_loss_list = []
    valid_loss_list = []

    for epoch in range(epoch_num):
        # Train model
        model.train()
        losses, nums = zip(*[loss_batch(model, loss_func, x, y, optimizer) for x, y in train_loader])
        train_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        train_loss_list.append(train_loss)

        # Validate model
        model.eval()
        with torch.no_grad():
            losses, corrects, nums = zip(*[valid_batch(model, loss_func, x, y) for x, y in valid_loader])
            valid_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
            valid_accuracy = np.sum(corrects) / np.sum(nums) * 100
            valid_loss_list.append(valid_loss)
            valid_accuracy_list.append(valid_accuracy)

            print(f"[Epoch {epoch+1}/{epoch_num}] "
                  f"Train loss: {train_loss:.3f}\t"
                  f"Validation loss: {valid_loss:.3f}\t",
                  f"Validation accruacy: {valid_accuracy:.2f}%")

    plt.figure()
    plt.plot(range(len(train_loss_list)), train_loss_list, label='Training Loss')
    plt.plot(range(len(valid_loss_list)), valid_loss_list, label='Validation Loss')
    plt.title(f'Training and Validation Loss Vs. Epoch Number ({epoch_num} Epochs)')
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.legend(loc="best")
    plt.savefig(f"images/Losses ({epoch_num} Epochs).png")
    plt.show()

    plt.figure()
    plt.plot(range(len(valid_accuracy_list)), valid_accuracy_list, label='Validation Accuracy')
    plt.title(f'Validation Accuracy Vs. Epoch Number ({epoch_num} Epochs)')
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.xlim([0, len(train_loss_list)])
    plt.ylim([0, 100])
    plt.legend(loc="best")
    plt.savefig(f"images/Accuracy ({epoch_num} Epochs).png")
    plt.show()