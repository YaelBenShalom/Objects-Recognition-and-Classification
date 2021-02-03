import os
import pickle

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
from read_data import ReadDataset
from run_model import run_model
from cnn import BasicNet

import matplotlib
import matplotlib.pyplot as plt

##### Loading Dataset #####
trainset_name = "train.p"
validset_name = "valid.p"
testset_name = "test.p"

train_features, train_labels = load_dataset(trainset_name)
valid_features, valid_labels = load_dataset(validset_name)
test_features, test_labels = load_dataset(testset_name)

# Getting  dataset properties
print("train_features shape: ", train_features.shape)
print("train_labels shape: ", train_labels.shape)
print("train dataset size: ", len(train_features))

print("valid_features: ", valid_features.shape)
print("valid_labels: ", valid_labels.shape)
print("Validation dataset size: ", len(valid_features))

print("test_features: ", test_features.shape)
print("test_labels: ", test_labels.shape)
print("Train dataset batch size: ", len(test_labels))

# Finiding the number of classes in the dataset
classes_num = len(set(train_labels))
print("Number of classes: ", classes_num)

# Finiding the size of the images in the dataset
image_shape = train_features[0].shape[:2]
print("images shape: ", image_shape)


##### Some visual aids #####
# Ploting class distribution for training set
fig, ax = plt.subplots()
ax.bar(range(classes_num), np.bincount(train_labels))
ax.set_title('Class Distribution in the Train Set')
ax.set_xlabel('Class Number')
ax.set_ylabel('Number of Events')
plt.show()

plt.figure(figsize=(12, 12))
for i in range(40):
    feature_index = random.randint(0, train_labels.shape[0])
    plt.subplot(6, 10, i+1)
    plt.axis('off')
    plt.imshow(train_features[feature_index])
plt.suptitle('Random Training Images')
plt.show()

plt.figure(figsize=(12, 12))
for i in range(classes_num):
    feature_index = random.choice(np.where(train_labels == i)[0])
    plt.subplot(6, 10, i+1)
    plt.axis('off')
    plt.title(f'class {i}')
    plt.imshow(train_features[feature_index])
plt.suptitle('Random training images from different classes')
plt.show()


# Computing data transformation to normalize data
mean = (0.485, 0.456, 0.406)    # from transforms.Compose example (https://pytorch.org/docs/stable/torchvision/transforms.html)
std = (0.229, 0.224, 0.225)     # -"-
transform = transforms.Compose([transforms.Resize((32, 32)),
                                # transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
                                transforms.ToTensor(), 
                                transforms.Normalize(mean=mean, std=std)])

train_dataset = ReadDataset(trainset_name, transform=transform)
valid_dataset = ReadDataset(validset_name, transform=transform)
test_dataset = ReadDataset(testset_name, transform=transform)

# # training set split
# training_dataset_size = [100]

# running_time_list = []
# train_accuracy_list = []
# valid_accuracy_list = []
# train_loss_list = []
# valid_loss_list = []


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


model = BasicNet()
learning_rate = 1e-5
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
device = torch.device('cpu')

# start_time = time.time()
model, train_loss, train_accuracy = run_model(model, running_mode='train', train_set=train_loader,
                                              valid_set=valid_loader, test_set=test_loader,
                                              batch_size=10, learning_rate=learning_rate,
                                              n_epochs=100, stop_thr=1e-4, shuffle=True)

print("train_loss is: ", train_loss)
print("train_accuracy is: ", train_accuracy)

# end_time = time.time()
# running_time = end_time - start_time
# running_time_list.append(running_time)

# train_loss_list.append(np.mean(train_loss['train']))
# valid_loss_list.append(np.mean(train_loss['valid']))
# train_accuracy_list.append(np.mean(train_accuracy['train']))
# valid_accuracy_list.append(np.mean(train_accuracy['valid']))

test_loss, test_accuracy = run_model(model, running_mode='test', train_set=train_loader,
                                     valid_set=valid_loader, test_set=test_loader,
                                     batch_size=10, learning_rate=learning_rate,
                                     n_epochs=100, stop_thr=1e-4, shuffle=True)

print("test_loss is: ", test_loss)
print("test_accuracy is: ", test_accuracy)