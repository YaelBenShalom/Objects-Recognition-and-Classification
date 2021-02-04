import numpy as np
import argparse

import torch
from torch import nn
from torchvision.utils import make_grid

from train import valid_batch

import matplotlib.pyplot as plt


def predict(model, data_loader, loss_func=nn.CrossEntropyLoss(), device=torch.device('cpu')):
    """
    This function either evaluates a model.
    """
    model.eval()
    with torch.no_grad():
        losses, corrects, nums = zip(*[valid_batch(model, loss_func, x, y) for x, y in data_loader])
        test_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        test_accuracy = np.sum(corrects) / np.sum(nums) * 100

        print(f"Test loss: {test_loss:.3f}")
        print(f"Test accruacy: {test_accuracy:.2f}%")