# Traffic-Sign Recognition

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

In this module, I trained a neural network to detect and classify different traffic signs using PyTorch, YoloV5 and OpenCV.<br>
I based my program on the German Traffic Sign Detection Benchmark ([GTSBB](https://benchmark.ini.rub.de/gtsdb_news.html)) dataset.

An example of output video:<br>
![Traffic-Sign Recognition](https://github.com/YaelBenShalom/Objects-Recognition-and-Classification/blob/master/traffic_signs_detection/traffic_signs_recognition/images/traffic-sign.gif?raw=true)

## User Guide

### Program Installation

1. Clone the repository, using the following commands:

   ```
   git clone https://github.com/YaelBenShalom/Objects-Recognition-and-Classification/tree/master/traffic_signs_detection/traffic_signs_recognition
   ```

2. Download the dataset and extract it into `./data` directory. The dataset can be found on the [INI Benchmark website](https://benchmark.ini.rub.de/?section=gtsrb&subsection=news), or downloaded directly through [here](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html).

3. Download the model and extract it into `./model` directory.

### Quickstart Guide

Run the recognition program:

1. To train the program on the dataset and test it on a specific video, copy the video into `./input_video` directory and run the following command from the root directory:
   ```
   python code/recognition_videos.py --video <video-name>
   ```
   Where `<video-name>` is the name of the video (including video type).

## Dataset

The German Traffic Sign Recognition Benchmark ([GTSRB](https://benchmark.ini.rub.de/gtsrb_news.html)) is a large multi-category classification benchmark. It was used in a competition at the International Joint Conference on Neural Networks (IJCNN) 2011.

The benchmark has the following properties:

1. Single-image, multi-class classification problem.
2. 43 classes.
3. More than 50,000 images in total (~35,000 training images, ~4000 validation images, and ~13,000 testing images).
4. Large, lifelike database.

The dataset can be found on the [INI Benchmark website](https://benchmark.ini.rub.de/?section=gtsrb&subsection=news), or downloaded directly through [here](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html).

![The German Traffic Sign Detection Benchmark](https://github.com/YaelBenShalom/Traffic-Sign-Recognition-and-Classification/blob/master/traffic_signs_detection/traffic_signs_recognition/images/dataset.png?raw=true)
