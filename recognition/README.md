# Traffic-Sign Recognition
## ME499 - Independent Project, Winter 2021
Yael Ben Shalom, Northwestern University.<br>
This module is a part of a [Traffic-Sign Recognition and Classification](https://github.com/YaelBenShalom/Traffic-Sign-Recognition-and-Classification) project.


## Module Overview
In this module I trained a neural network to detect and classify different traffic signs using YOLOv3 and OpenCV.<br>
In this project, I used the German Traffic Sign Recognition Benchmark ([GTSRB](https://benchmark.ini.rub.de/gtsrb_news.html)) dataset.


## User Guide
### Program Installation

1. Clone the repository, using the following commands:
    ```
    git clone https://github.com/YaelBenShalom/Traffic-Sign-Recognition-and-Classification/recognition
    ```

2. Download the dataset and extract it into `./data` directory. The dataset can be found on the [INI Benchmark website](https://benchmark.ini.rub.de/?section=gtsrb&subsection=news), or downloaded directly through [here](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html).

3. Download the model and extract it into `./model` directory. The dataset can be found on the [INI Benchmark website](https://benchmark.ini.rub.de/?section=gtsrb&subsection=news), or downloaded directly through [here](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html).

### Quickstart Guide

Run the recognition program:
1. To train the program on the dataset and test it on a specific video, copy the video into `./input_video` directory and run the following command from the root directory:
    ```
    python code/recognition.py --video <video-name>
    ```
    Where `<video-name>` is the name of the video (including video type).



## Project Architecture
...


## Dataset
...
