import os
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn import ReLU, Linear
from torch import Tensor

import matplotlib.pyplot as plt


# Finding the datased directories path:
dataset_path = "../GTSRB-German_Traffic_Sign_Recognition_Benchmark"
dataset_dir = os.listdir(dataset_path)
print("dataset directory contain: ", dataset_dir)

meta_path = "../GTSRB-German_Traffic_Sign_Recognition_Benchmark/Meta"
meta_info_path = os.path.join(dataset_path, 'Meta.csv')

train_dataset_path = "../GTSRB-German_Traffic_Sign_Recognition_Benchmark/Train"
train_info_path = os.path.join(dataset_path, 'Train.csv')
classes = os.listdir(train_dataset_path)        # Dataset classes
classes_num = len(classes)                      # Number of classes
print("The classes: ", classes)

test_dataset_path = "../GTSRB-German_Traffic_Sign_Recognition_Benchmark/Test"
test_info_path = os.path.join(dataset_path, 'Test.csv')

# Declaring dataset labels (in the same order as the Meta directory):
labels = ['20km/h limit', '30km/h limit', '50km/h limit', '60km/h limit', '70km/h limit', '80km/h limit', 'End of 80km/h limit', '100km/h limit', '120km/h limit',
          'No overtaking', 'No overtaking (tracks)', 'Upcoming intersection priority', 'Priority road', 'Give way', 'Stop', 'No entry (vehicles)', 'No entry (tracks)', 'No entry',
          'General danger', 'Left curve', 'Right curve', 'Double curve', 'Rough road', 'Slippery road', 'Road narrows', 'Work in process', 'Traffic light',
          'Pedestrian crossing', 'Children', 'Cyclists', 'Icy road', 'Wild animals', 'End of overtaking and speed limits', 'Only right', 'Only left', 'Only straight',
          'Only straight and right', 'Only straight and left', 'Drive on right side', 'Drive on left side', 'Traffic circle', 'End of no overtaking', 'End of no overtaking (tracks)']


# transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform = transforms.Compose([transforms.RandomCrop(32), 
                                 transforms.RandomHorizontalFlip(), 
                                 transforms.ToTensor(), 
                                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                           shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                          shuffle=False, num_workers=2)

# Defining training and validation sets:
train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=transform)
train_dataset_len = len(train_dataset)
print("Train dataset size: ", train_dataset_len)

train_dataset ,valid_dataset = torch.utils.data.random_split(train_dataset, [math.ceil(0.75*train_dataset_len), math.floor(0.25*train_dataset_len)])

