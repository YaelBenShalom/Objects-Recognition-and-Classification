import argparse
import os
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from load_data import load_dataset
from read_data import ReadDataset
from run_model import run_model, predict
from cnn import BaselineNet, ResNet


def load_arguments(args):
    """
    This function loads the image argument.
    If an image was added as argument, the program will predict its class after training and testing the model.
    If no image was added as argument, the program will only train and test the results on the datasets.
    Inputs:
      args:                 the input arguments of the program in the form of a dictionary {"image" : <argument>}.
                            if args exist, <argument> is the input image, else <argument> is None.
    Output:
      test_image:           the input image that should be tested by the model.
    """
    test_image = plt.imread(args["image"])
    plt.figure()
    plt.imshow(test_image)
    plt.show()

    return test_image


def dataset_properties(trainset_name, validset_name, testset_name, class_names, data_dir):
    """
    This function finds the dataset properties.
    This function is for information only.

    Inputs:
      trainset_name:        the name of the training set file.
      validset_name:        the name of the validation set file.
      testset_name:         the name of the testing set file.

    Output: None
    """
    train_features, train_labels = load_dataset(
        trainset_name, base_folder=data_dir)
    valid_features, valid_labels = load_dataset(
        validset_name, base_folder=data_dir)
    test_features, test_labels = load_dataset(
        testset_name, base_folder=data_dir)

    # print(f"train_features shape: {train_features.shape}")
    # print(f"train_labels shape: {train_labels.shape}")
    print(f"train dataset size: {len(train_features)}")

    # print(f"valid_features: {valid_features.shape}")
    # print(f"valid_labels: {valid_labels.shape}")
    print(f"validation dataset size: {len(valid_features)}")

    # print(f"test_features: {test_features.shape}")
    # print(f"test_labels: {test_labels.shape}")
    print(f"test dataset size: {len(test_labels)}")

    # Finding the number of classes in the dataset
    classes_num = len(set(train_labels))
    print(f"Number of classes: {classes_num}")

    # Finding the size of the images in the dataset
    image_shape = train_features[0].shape[:2]
    print(f"images shape: {image_shape}")

    # Plotting class distribution for training set
    fig, ax = plt.subplots()
    ax.bar(range(classes_num), np.bincount(train_labels))
    ax.set_title('Class Distribution in the Train Set', fontsize=20)
    ax.set_xlabel('Class Number')
    ax.set_ylabel('Number of Events')
    plt.savefig('images/Class_Distribution.png')
    plt.show()

    # Plotting random 40 images from train set
    plt.figure(figsize=(12, 12))
    for i in range(40):
        feature_index = random.randint(0, train_labels.shape[0])
        plt.subplot(6, 8, i + 1)
        plt.subplots_adjust(left=0.1, bottom=0.03, right=0.9,
                            top=0.92, wspace=0.2, hspace=0.2)
        plt.axis('off')
        plt.imshow(train_features[feature_index])
    plt.suptitle('Random Training Images', fontsize=20)
    plt.savefig('images/Random_Training_Images.png')
    plt.show()

    # Plotting images for every class from train set
    plt.figure(figsize=(14, 14))
    for i in range(classes_num):
        feature_index = random.choice(np.where(train_labels == i)[0])
        plt.subplot(6, 8, i + 1)
        plt.subplots_adjust(left=0.1, bottom=0.03, right=0.9,
                            top=0.92, wspace=0.2, hspace=0.2)
        plt.axis('off')
        plt.title(class_names[i], fontsize=10)
        plt.imshow(train_features[feature_index])
    plt.suptitle('Random training images from different classes', fontsize=20)
    plt.savefig('images/Random_Training_Images_Different_Class.png')
    plt.show()


def class_names_fun(data_dir):
    """
    This function returns a dictionary with the classes numbers and names.

    Inputs: None

    Output:
      class_names:          a dictionary with the classes numbers and names.
    """
    # Class names path
    class_names_path = os.path.join(data_dir, "label_names.csv")
    class_names_rows = open(class_names_path).read().strip().split("\n")[1:]

    # Defining class names dictionary
    class_names = {}
    for row in class_names_rows:
        label, label_name = row.strip().split(",")
        class_names[int(label)] = label_name

    return class_names


def plot_training_results(train_loss_list, valid_loss_list, valid_accuracy_list, epoch_num):
    """
    This function plots the results of training the network.

    Inputs:
      train_loss_list:      list of loss value on the entire training dataset.
      valid_loss_list:      list of loss value on the entire validation dataset.
      valid_accuracy_list:  list of accuracy on the entire validation dataset.

    Output: None
    """
    # Plotting training and validation loss vs. epoch number
    plt.figure()
    plt.plot(range(len(train_loss_list)),
             train_loss_list, label='Training Loss')
    plt.plot(range(len(valid_loss_list)),
             valid_loss_list, label='Validation Loss')
    plt.title(
        f'Training and Validation Loss Vs. Epoch Number ({epoch_num} Epochs)')
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.legend(loc="best")
    plt.savefig(f"images/Losses ({epoch_num} Epochs).png")
    plt.show()

    # Plotting validation accuracy vs. epoch number
    plt.figure()
    plt.plot(range(len(valid_accuracy_list)),
             valid_accuracy_list, label='Validation Accuracy')
    plt.title(f'Validation Accuracy Vs. Epoch Number ({epoch_num} Epochs)')
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.xlim([0, len(train_loss_list)])
    plt.ylim([0, 100])
    plt.legend(loc="best")
    plt.savefig(f"images/Accuracy ({epoch_num} Epochs).png")
    plt.show()


def main(args):
    """ Main function of the program
    Inputs:
      args:                 the input arguments of the program in the form of a dictionary {"image" : <argument>}.
                            if args exist, <argument> is the input image, else <argument> is None.
    Output: None
    """

    # Define dataset directory
    data_dir = "data"

    # Define dataset files
    trainset_name = "train.p"
    validset_name = "valid.p"
    testset_name = "test.p"

    # Finding dataset properties
    class_names = class_names_fun(data_dir)

    # Visualizing the dataset
    dataset_properties(trainset_name, validset_name,
                       testset_name, class_names, data_dir)

    # Define the device parameters
    torch.manual_seed(1)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Define the model
    model = BaselineNet().to(device)

    # Define the training properties
    epoch_num = 100
    criterion = nn.CrossEntropyLoss()
    learning_rate = 1e-3
    batch_size = 64
    stop_threshold = 1e-4

    # Computing data transformation to normalize data
    # from https://pytorch.org/docs/stable/torchvision/transforms.html
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)     # -"-
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((32, 32)),
                                    transforms.Normalize(mean=mean, std=std)])

    # Reading the datasets
    train_dataset = ReadDataset(trainset_name, transform=transform)
    valid_dataset = ReadDataset(validset_name, transform=transform)
    test_dataset = ReadDataset(testset_name, transform=transform)

    # If no input model - training a new model
    if not args["model"]:
        # Defining the model
        model_path = os.path.abspath("model")

        # Train the network
        model, train_loss_list, valid_loss_list, valid_accuracy_list = run_model(model, running_mode='train',
                                                                                 train_set=train_dataset,
                                                                                 valid_set=valid_dataset,
                                                                                 test_set=test_dataset,
                                                                                 batch_size=batch_size, epoch_num=epoch_num,
                                                                                 learning_rate=learning_rate,
                                                                                 stop_thr=stop_threshold,
                                                                                 criterion=criterion, device=device)
        # Plot the results of training the network
        plot_training_results(train_loss_list, valid_loss_list,
                              valid_accuracy_list, epoch_num)

        # Save the trained model
        torch.save(model.state_dict(), model_path)

    # If input model - load the existing model
    else:
        # Defining the model
        model_path = os.path.abspath(args["model"])

        # Load the trained model
        model.load_state_dict(torch.load(model_path, map_location=device))

    # Test the network
    test_loss, test_accuracy = run_model(model, running_mode='test', train_set=train_dataset,
                                         valid_set=valid_dataset, test_set=test_dataset,
                                         batch_size=batch_size, epoch_num=epoch_num,
                                         learning_rate=learning_rate, stop_thr=stop_threshold,
                                         criterion=criterion, device=device)

    print(f"Test loss: {test_loss:.3f}")
    print(f"Test accuracy: {test_accuracy:.2f}%")

    # Check if image argument exists
    if args["image"]:
        # Load the image argument
        test_image = load_arguments(args)
        test_image_resized = cv2.resize(test_image, (32, 32))

        test_image_tensor = transforms.ToTensor()(np.array(test_image_resized))

        # Transform tested image
        test_image_transform4d = test_image_tensor.unsqueeze(0)

        # Predict the class of the tested image
        prediction = int(predict(model, test_image_transform4d)[0])
        print(
            f"Test prediction: {prediction} -> Class: {class_names[prediction]}")

        # Plot the image with the predicted class
        plt.figure()
        plt.axis('off')
        plt.title(class_names[prediction], fontsize=10)
        plt.imshow(test_image)
        plt.suptitle('Image Classification', fontsize=18)
        plt.savefig('images/Image_Classification')
        plt.show()


if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", help="path to the input image")
    parser.add_argument("-m", "--model", help="path to the input image")
    args = vars(parser.parse_args())
    main(args)
