# Trash Recognition

## ME499 - Independent Project, Winter 2021

Yael Ben Shalom, Northwestern University.<br>
This module is a part of a [Objects Recognition and Classification](https://github.com/YaelBenShalom/Objects-Recognition-and-Classification) project.

## Table of Contents

- [Module Overview](#module-overview)
- [User Guide](#user-guide)
  - [Program Installation](#program-installation)
  - [Data Preprocess](#data-preprocess)
  - [Training the Model with YoloV5](#training-the-model-with-yolov5)
  - [Recognizing Recyclable Objects](#recognizing-recyclable-objects)
- [Dataset](#dataset)

## Module Overview

In this module I trained a neural network to detect and classify different recyclable objects using PyTorch, YoloV5 and OpenCV.<br>
I based my program on the Trash Annotations in Context ([TACO](http://tacodataset.org/)) dataset.<br>
The TACO dataset contains ~60 different classes, but in this project I only detect 10 different objects.

An example of output video:<br>
![Trash Recognition](https://github.com/YaelBenShalom/Objects-Recognition-and-Classification/blob/master/trash_detection/trash_recognition/images/real-time%20detection2.gif)

## User Guide

### Program Installation

1. Clone the repository, using the following commands:

   ```
   git clone https://github.com/YaelBenShalom/Objects-Recognition-and-Classification/tree/master/trash_recognition/trash_recognition
   ```

2. Download the dataset and extract it into `./data` directory. The dataset can be found on the [TACO dataset website](http://tacodataset.org/), or downloaded directly through [here](https://www.kaggle.com/kneroma/tacotrashdataset/download).

3. Clone the yolov5 repository, using the following commands:
   ```
   git clone https://github.com/rkuo2000/yolov5
   ```

### Data Preprocess

1. Make `tmp/labels/` and `tmp/images/` folders in the root directory.

2. Run the `code/preprocess_data.py` script. the script converts the dataset to Yolo format (converts .json files to .txt files, and splits the dataset to train, validation and test folders).

### Training the Model with YoloV5

1. Copy the `taco10.yaml` file into `/yolov5/models/` directory.

2. Train the model using the following command:

   ```
   python train.py --img 320 --batch 1 --epochs 100 --data models/taco10.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt
   ```

   **Beware!** It might take a while...

3. Copy the trained weight into the `/weights` folder using the following command:

   ```
   cp runs/exp16/weights/best.pt weights
   ```

4. Test the trained model on the test-set using the following command:
   ```
   python detect.py --weights weights/best.pt --img 320 --conf 0.4 --source taco/test/images
   ```
   Or test it on a specific image using the following command:
   ```
   python detect.py --weights weights/best.pt --img 320 --conf 0.4 --source taco/test/images/<image-name>
   ```
   Where `<image-name>` is the name of the image or video (including image type).<br>
   The detected images/videos located in `inference/output`.

### Recognizing Recyclable Objects

After training the model, run the recognition program:

1. To recognizing trash in a specific video, copy the video into `./input_video` directory and run the following command from the root directory:

   ```
   python code/recognition_videos.py --video <video-name>
   ```

   Where `<video-name>` is the name of the video (including video type).

   ![Trash Recognition](https://github.com/YaelBenShalom/Objects-Recognition-and-Classification/blob/master/trash_detection/trash_recognition/images/detecting_baxter.gif)

## Dataset

TACO is a growing image dataset of waste in the wild. It contains images of litter taken under diverse environments: woods, roads and beaches. These images are manually labeled and segmented according to a hierarchical taxonomy to train and evaluate object detection algorithms.<br>
The dataset currently contain 60 different classes.<br>
For convenience, annotations are provided in COCO format.

For more information about the TACO dataset, check out the dataset's [website](http://tacodataset.org/) or the [paper](https://arxiv.org/abs/2003.06975) written about the dataset and the trash annotation project.

For download instruction, check out the dataset's [GitHub page](https://github.com/pedropro/TACO).
