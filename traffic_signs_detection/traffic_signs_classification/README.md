# Traffic-Sign Classification

## ME499 - Independent Project, Winter 2021

Yael Ben Shalom, Northwestern University.<br>
This module is a part of a [Objects Recognition and Classification](https://github.com/YaelBenShalom/Objects-Recognition-and-Classification) project.

## Table of Contents

- [Module Overview](#module-overview)
- [User Guide](#user-guide)
  - [Program Installation](#program-installation)
  - [Quickstart Guide](#quickstart-guide)
- [Dataset](#dataset)

## Module Overview

In this module I built and trained a neural network to classify different traffic signs using PyTorch.<br>
I based my program on the German Traffic Sign Recognition Benchmark ([GTSRB](https://benchmark.ini.rub.de/gtsrb_news.html)) dataset.

## User Guide

### Program Installation

1. Clone the repository, using the following commands:

   ```
   git clone https://github.com/YaelBenShalom/Objects-Recognition-and-Classification/tree/master/traffic_signs_detection/traffic_signs_classification
   ```

2. Download the dataset and extract it into `./data` directory. The dataset can be found on the [INI Benchmark website](https://benchmark.ini.rub.de/?section=gtsrb&subsection=news), or downloaded directly through [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip).

### Quickstart Guide

Run the classification program:

1. To train and test the program on the dataset, run the following command from the root directory:

   ```
   python code/classification.py
   ```

2. To train the program on the dataset and test it on a specific image, copy the image to the root directory and run the following command from the root directory:

   ```
   python code/classification.py --image <image-name>
   ```

   Where `<image-name>` is the name of the image (including image type).<br>
   The trained model will be saved in the root directory as `/model`.

3. To to use an existing model and test it on a specific image, copy the image to the root directory and run the following command from the root directory:
   ```
   python code/classification.py --image <image-name> --model <model-name>
   ```
   Where `<model-name>` is the name of the trained model.

<br>The program output when running it on the example image:

The loss plot:<br>
![Loss Graph](<https://github.com/YaelBenShalom/Objects-Recognition-and-Classification/blob/master/traffic_signs_detection/traffic_signs_classification/images/Losses%20(100%20Epochs).png>)

The accuracy plot:<br>
![Accuracy Graph](<https://github.com/YaelBenShalom/Objects-Recognition-and-Classification/blob/master/traffic_signs_detection/traffic_signs_classification/images/Accuracy%20(100%20Epochs).png>)

The output image (with the correct prediction):<br>
![Accuracy Graph](https://github.com/YaelBenShalom/Objects-Recognition-and-Classification/blob/master/traffic_signs_detection/traffic_signs_classification/images/Image_Classification.png)

## Dataset

The German Traffic Sign Recognition Benchmark ([GTSRB](https://benchmark.ini.rub.de/gtsrb_news.html)) is a large multi-category classification benchmark. It was used in a competition at the International Joint Conference on Neural Networks (IJCNN) 2011.

The benchmark has the following properties:

1. Single-image, multi-class classification problem.
2. 43 classes.
3. More than 50,000 images in total (~35,000 training images, ~4000 validation images, and ~13,000 testing images).
4. Large, lifelike database.

The dataset can be found on the [INI Benchmark website](https://benchmark.ini.rub.de/?section=gtsrb&subsection=news), or downloaded directly through [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip).

![The German Traffic Sign Recognition Benchmark](https://github.com/YaelBenShalom/Traffic-Sign-Recognition-and-Classification/blob/master/traffic_signs_detection/traffic_signs_classification/images/dataset.png)
