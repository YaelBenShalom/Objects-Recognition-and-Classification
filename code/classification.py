import os
import pickle
import PIL

import numpy as np
import random
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn import ReLU, Linear
from torch import Tensor

from load_data import load_dataset
from read_data import ReadDataset, CustomDataLoader
from run_model import run_model
from cnn import BasicNet
from train import fit
from test import predict
import matplotlib
import matplotlib.pyplot as plt

##### Loading Dataset #####
trainset_name = "train.p"
validset_name = "valid.p"
testset_name = "test.p"

train_features, train_labels = load_dataset(trainset_name, base_folder='data')
valid_features, valid_labels = load_dataset(validset_name, base_folder='data')
test_features, test_labels = load_dataset(testset_name, base_folder='data')

# Getting  dataset properties
print(f"train_features shape: {train_features.shape}")
print(f"train_labels shape: {train_labels.shape}")
print(f"train dataset size: {len(train_features)}")

print(f"valid_features: {valid_features.shape}")
print(f"valid_labels: {valid_labels.shape}")
print(f"Validation dataset size: {len(valid_features)}")

print(f"test_features: {test_features.shape}")
print(f"test_labels: {test_labels.shape}")
print(f"Train dataset batch size: {len(test_labels)}")

# Finiding the number of classes in the dataset
classes_num = len(set(train_labels))
print(f"Number of classes: {classes_num}")

# Finiding the size of the images in the dataset
image_shape = train_features[0].shape[:2]
print(f"images shape: {image_shape}")


# ##### Some visual aids #####
# # Ploting class distribution for training set
# fig, ax = plt.subplots()
# ax.bar(range(classes_num), np.bincount(train_labels))
# ax.set_title('Class Distribution in the Train Set')
# ax.set_xlabel('Class Number')
# ax.set_ylabel('Number of Events')
# plt.show()

# plt.figure(figsize=(12, 12))
# for i in range(40):
#     feature_index = random.randint(0, train_labels.shape[0])
#     plt.subplot(6, 10, i+1)
#     plt.axis('off')
#     plt.imshow(train_features[feature_index])
# plt.suptitle('Random Training Images')
# plt.show()

# plt.figure(figsize=(12, 12))
# for i in range(classes_num):
#     feature_index = random.choice(np.where(train_labels == i)[0])
#     plt.subplot(6, 10, i+1)
#     plt.axis('off')
#     plt.title(f'class {i}')
#     plt.imshow(train_features[feature_index])
# plt.suptitle('Random training images from different classes')
# plt.show()


# Computing data transformation to normalize data
mean = (0.485, 0.456, 0.406)    # from transforms.Compose example (https://pytorch.org/docs/stable/torchvision/transforms.html)
std = (0.229, 0.224, 0.225)     # -"-
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize(mean=mean, std=std)])

train_dataset = ReadDataset(trainset_name, transform=transform)
valid_dataset = ReadDataset(validset_name, transform=transform)
test_dataset = ReadDataset(testset_name, transform=transform)


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_device(x, y):
    return x.to(device), y.to(device, dtype=torch.int64)

train_loader = CustomDataLoader(train_loader, to_device)
valid_loader = CustomDataLoader(valid_loader, to_device)
test_loader = CustomDataLoader(test_loader, to_device)


model = BasicNet().to(device)
epoch_num = 100
criterion = nn.CrossEntropyLoss()
learning_rate = 1e-3

# Train model
fit(model, train_loader, valid_loader, epoch_num, learning_rate=learning_rate,
    loss_func=criterion, device = device)

# Test model
predict(model, test_loader, loss_func=criterion, device = device)












# model, train_loss, train_accuracy = run_model(model, running_mode='train', train_set=train_dataset,
#                                               valid_set=valid_dataset, test_set=test_dataset,
#                                               batch_size=10, learning_rate=learning_rate,
#                                               epoch_num=100, criterion=criterion, stop_thr=1e-4,
#                                               shuffle=True)

# print("train_loss is: ", train_loss)
# print("train_accuracy is: ", train_accuracy)

# test_loss, test_accuracy = run_model(model, running_mode='test', train_set=train_dataset,
#                                      valid_set=valid_dataset, test_set=test_dataset,
#                                      batch_size=10, learning_rate=learning_rate,
#                                      epoch_num=100, criterion=criterion,
#                                      stop_thr=1e-4, shuffle=True)

# print("test_loss is: ", test_loss)
# print("test_accuracy is: ", test_accuracy)