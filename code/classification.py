import os
import pickle
# import PIL
import argparse

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
from torch import Tensor

from load_data import load_dataset
from read_data import ReadDataset
from run_model import run_model, predict
from cnn import BaselineNet

import matplotlib
import matplotlib.pyplot as plt
import cv2
##### Loading Dataset #####
trainset_name = "train.p"
validset_name = "valid.p"
testset_name = "test.p"

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the input image")
args = vars(ap.parse_args())
print(f"train_features shape: {args}")

if args["image"]:
    test_image = cv2.imread(args["image"])
    print(f"train_features shape: {test_image.shape}")
    test_image = cv2.resize(test_image, (32, 32))
    print(f"train_features shape: {test_image.shape}")

train_features, train_labels = load_dataset(trainset_name, base_folder='data')
valid_features, valid_labels = load_dataset(validset_name, base_folder='data')
test_features, test_labels = load_dataset(testset_name, base_folder='data')

# Finiding dataset properties
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

# class names dictionary
class_names = { 0: '20km/h limit',      
                1: '30km/h limit',
                2: '50km/h limit',
                3: '60km/h limit',
                4: '70km/h limit',
                5: '80km/h limit',
                6: 'End of 80km/h limit',
                7: '100km/h limit',
                8: '120km/h limit',
                9: 'No overtaking',
                10: 'No overtaking (tracks)',
                11: 'Intersection priority',
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
                32: 'End of limits',
                33: 'Only right',
                34: 'Only left',
                35: 'Only straight',
                36: 'Only straight/right',
                37: 'Only straight/left',
                38: 'Drive on right side',
                39: 'Drive on left side',
                40: 'Traffic circle',
                41: 'End of no overtaking',
                42: 'End of no overtaking (tracks)' }

# ##### Some visual aids #####
# # Ploting class distribution for training set
# fig, ax = plt.subplots()
# ax.bar(range(classes_num), np.bincount(train_labels))
# ax.set_title('Class Distribution in the Train Set', fontsize=20)
# ax.set_xlabel('Class Number')
# ax.set_ylabel('Number of Events')
# plt.savefig('images/Class_Distribution.png')
# plt.show()

# Ploting random 40 images from train set
# plt.figure(figsize=(12, 12))
# for i in range(40):
#     feature_index = random.randint(0, train_labels.shape[0])
#     plt.subplot(6, 8, i+1)
#     plt.subplots_adjust(left=0.1, bottom=0.03, right=0.9, top=0.92, wspace=0.2, hspace=0.2)
#     plt.axis('off')
#     plt.imshow(train_features[feature_index])
# plt.suptitle('Random Training Images', fontsize=20)
# plt.savefig('images/Random_Training_Images.png')
# plt.show()

# # Ploting images for every class from train set
# plt.figure(figsize=(14, 14))
# for i in range(classes_num):
#     feature_index = random.choice(np.where(train_labels == i)[0])
#     plt.subplot(6, 8, i+1)
#     plt.subplots_adjust(left=0.1, bottom=0.03, right=0.9, top=0.92, wspace=0.2, hspace=0.2)
#     plt.axis('off')
#     # plt.title(f'class {i}')
#     plt.title(class_names[i], fontsize=10)
#     plt.imshow(train_features[feature_index])
# plt.suptitle('Random training images from different classes', fontsize=20)
# plt.savefig('images/Random_Training_Images_Different_Class.png')
# plt.show()

torch.manual_seed(1)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model = BaselineNet().to(device)
epoch_num = 1
criterion = nn.CrossEntropyLoss()
learning_rate = 1e-3
batch_size = 64
stop_threshold = 1e-4

# Computing data transformation to normalize data
mean = (0.485, 0.456, 0.406)    # from transforms.Compose example (https://pytorch.org/docs/stable/torchvision/transforms.html)
std = (0.229, 0.224, 0.225)     # -"-
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize(mean=mean, std=std)])

# # Reading the datasets
train_dataset = ReadDataset(trainset_name, transform=transform)
valid_dataset = ReadDataset(validset_name, transform=transform)
test_dataset = ReadDataset(testset_name, transform=transform)
test_image_transform = transform(test_image)
# print(f"test_image_transform: {test_image_transform}")

# Train model
model, train_loss_list, valid_loss_list, valid_accuracy_list = run_model(model, running_mode='train', train_set=train_dataset,
                                                          valid_set=valid_dataset, test_set=test_dataset,
                                                          batch_size=batch_size, epoch_num=epoch_num, 
                                                          learning_rate=learning_rate, stop_thr=stop_threshold,
                                                          criterion=criterion, device=device, shuffle=True)

# Test model
test_loss, test_accuracy = run_model(model, running_mode='test', train_set=train_dataset,
                                     valid_set=valid_dataset, test_set=test_dataset,
                                     batch_size=batch_size, epoch_num=epoch_num, 
                                     learning_rate=learning_rate, stop_thr=stop_threshold,
                                     criterion=criterion, device=device, shuffle=True)

test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
test_loader = CustomDataLoader(test_loader, to_device)
prediction = predict(model, test_image_transform)
print(f"Test prediction: {prediction}")

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