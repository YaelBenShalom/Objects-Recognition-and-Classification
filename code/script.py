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

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.facecolor'] = '#ffffff' #TODO


# Finding the datased directories path:
dataset_path = "../Dataset-Traffic-Detection"
dataset_dir = os.listdir(dataset_path)
print("dataset directory contains: ", dataset_dir)

meta_path = "../Dataset-Traffic-Detection/Meta"
meta_info_path = os.path.join(dataset_path, 'Meta.csv')

train_dataset_path = "../Dataset-Traffic-Detection/Train"
train_info_path = os.path.join(dataset_path, 'Train.csv')

test_dataset_path = "../Dataset-Traffic-Detection/Test"
test_info_path = os.path.join(dataset_path, 'Test.csv')

classes = os.listdir(train_dataset_path)        # Dataset classes
classes_num = len(classes)                      # Number of classes
print("Number of classes: ", classes_num)

class_names = { 0: '20km/h speed limit',              # classes names (in the same order as the Meta directory)
                1: '30km/h speed limit',
                2: '50km/h speed limit',
                3: '60km/h speed limit',
                4: '70km/h speed limit',
                5: '80km/h speed limit',
                6: 'End of 80km/h speed limit',
                7: '100km/h speed limit',
                8: '120km/h speed limit',
                9: 'No overtaking',
                10: 'No overtaking (tracks)',
                11: 'Upcoming intersection priority',
                12: 'Priority road',
                13: 'Give way',
                14: 'Stop',
                15: 'No entry (vehicles)',
                16: 'No entry (tracks)',
                17: 'No entry',
                18: 'General danger',
                19: 'Left curve',
                20: 'Right curve',
                21: 'Double curve',
                22: 'Rough road',
                23: 'Slippery road',
                24: 'Road narrows',
                25: 'Work in process',
                26: 'Traffic light',
                27: 'Pedestrian crossing',
                28: 'Children',
                29: 'Cyclists',
                30: 'Icy road',
                31: 'Wild animals',
                32: 'End of overtaking and speed limits',
                33: 'Only right',
                34: 'Only left',
                35: 'Only straight',
                36: 'Only straight and right',
                37: 'Only straight and left',
                38: 'Drive on right side',
                39: 'Drive on left side',
                40: 'Traffic circle',
                41: 'End of no overtaking',
                42: 'End of no overtaking (tracks)' }

# transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

mean = (0.485, 0.456, 0.406)    # from transforms.Compose example (https://pytorch.org/docs/stable/torchvision/transforms.html)
std = (0.229, 0.224, 0.225)     # -"-

# transform = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'), # TODO
#                                  transforms.RandomHorizontalFlip(), 
#                                  transforms.ToTensor(), 
#                                  transforms.Normalize(mean=mean, std=std)])

transform = transforms.Compose([transforms.Resize((32, 32)),
                                # transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
                                transforms.ToTensor(), 
                                transforms.Normalize(mean=mean, std=std)])

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

train_dataset ,valid_dataset = torch.utils.data.random_split(train_dataset, [math.ceil(0.9*train_dataset_len), math.floor(0.1*train_dataset_len)])

# test_dataset = torchvision.datasets.ImageFolder(root=test_dataset_path, transform=transform)

print("Train dataset size: ", len(train_dataset))
print("Validation dataset size: ", len(valid_dataset))
# print("Test dataset size: ", len(test_dataset))

train_example_batch = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True,
                                num_workers=3, pin_memory=True)
print("Train dataset batch size: ", len(train_example_batch))

means = torch.tensor(mean).reshape(1, 3, 1, 1)
stds = torch.tensor(std).reshape(1, 3, 1, 1)


for images, labels in train_example_batch:
    # print("images", images.shape)
    # print("labels", labels.shape)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xticks([]); ax.set_yticks([])
    images = images * stds + means
    ax.imshow(torchvision.utils.make_grid(images[:64], nrow=8).permute(1, 2, 0).clamp(0,1))
    break