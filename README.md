# Objects Recognition and Classification

Independent Project (ME499), Northwestern University, Winter 2021.<br>
Take a look at my [portfolio post](https://yaelbenshalom.github.io/ML_object_detection/index.html) for more information about the project.


## Table of Contents

- [Project Overview](#project-overview)
  - [Traffic sign detection and classification](#traffic-sign-detection-and-classification)
  - [Trash detection, classification, and segmentation](#trash-detection-classification-and-segmentation)
  - [Recycling Baxter Implementation](#recycling-baxter-implementation)

## Project Overview

This project contains 2 modules:

### Traffic sign detection and classification

In this module I built and trained a neural network to detect and classify different traffic signs using PyTorch, OpenCV and YoloV3.<br>
This module contained 2 parts:
- [Traffic sign classification](https://github.com/YaelBenShalom/Objects-Recognition-and-Classification/tree/master/traffic_signs_detection/traffic_signs_classification) - Were I build and trained a neural network to classify different traffic-signs images.
- [Traffic sign detection](https://github.com/YaelBenShalom/Objects-Recognition-and-Classification/tree/master/traffic_signs_detection/traffic_signs_recognition) - Were I trained neural network to detect different traffic-signs in an image/video.

An example of traffic detection program output:<br>
<p align="center">
  <img align="center" src="https://github.com/YaelBenShalom/Objects-Recognition-and-Classification/blob/master/traffic_signs_detection/traffic_signs_recognition/images/traffic-sign.gif">
</p>


### Trash detection, classification, and segmentation

In this module I built and trained a neural network to detect and classify different traffic signs using PyTorch, OpenCV and YoloV5.<br>
This module contained 2 parts:
- [Trash classification](https://github.com/YaelBenShalom/Objects-Recognition-and-Classification/tree/master/trash_detection/trash_classification) - Were I build and trained a neural network to classify different recyclable objects' images.
- [Trash detection](https://github.com/YaelBenShalom/Objects-Recognition-and-Classification/tree/master/trash_detection/trash_recognition) - Were I trained neural network to detect different recyclable objects in an image/video.

An example of trash detection program output:<br>
<p align="center">
  <img align="center" src="https://github.com/YaelBenShalom/Objects-Recognition-and-Classification/blob/master/trash_detection/trash_recognition/images/real-time%20detection2.gif">
</p>

### Recycling Baxter Implementation
As an additional development of the [Recycler Baxter](https://github.com/YaelBenShalom/Recycler-Baxter) project, I used the ML algorithm I implemented in this project to detect and locate the recyclable objects sorted by the baxter robot:
<p align="center">
  <img align="center" src="https://github.com/YaelBenShalom/Objects-Recognition-and-Classification/blob/master/trash_detection/trash_recognition/images/detecting_baxter.gif">
</p>
