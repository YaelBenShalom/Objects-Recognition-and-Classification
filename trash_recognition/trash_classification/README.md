# Trash Classification
## ME499 - Independent Project, Winter 2021
Yael Ben Shalom, Northwestern University.<br>
This module is a part of a [Traffic-Sign Recognition and Classification](https://github.com/YaelBenShalom/Traffic-Sign-Recognition-and-Classification) project.


## Module Overview
In this module I built and trained a neural network to classify different traffic signs using PyTorch.<br>
In this project, I used the [Garbage Classification Dataset](https://www.kaggle.com/asdasdasasdas/garbage-classification).


## User Guide
### Program Installation

1. Clone the repository, using the following commands:
    ```
    git clone https://github.com/YaelBenShalom/Traffic-Sign-Recognition-and-Classification/tree/master/trash_recognition/trash_classification
    ```

2. Extract the dataset located in the `./data` directory.


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
    Where `<model-name>` is the name of the trained model.<br>


<br>The program output when running it on the example image:

The loss plot:<br>
        ![Loss Graph](https://github.com/YaelBenShalom/Traffic-Sign-Recognition-and-Classification/blob/master/trash_recognition/trash_classification/images/Losses%20(100%20Epochs).png)


The accuracy plot:<br>
        ![Accuracy Graph](https://github.com/YaelBenShalom/Traffic-Sign-Recognition-and-Classification/blob/master/trash_recognition/trash_classification/images/Accuracy%20(100%20Epochs).png)


The output image (with the correct prediction):<br>
        ![Accuracy Graph](https://github.com/YaelBenShalom/Traffic-Sign-Recognition-and-Classification/blob/master/trash_recognition/trash_classification/images/Image_Classification.png)


## Dataset

The Garbage Classification Dataset contains 6 classifications: cardboard (393), glass (491), metal (400), paper(584), plastic (472), and trash(127).

The dataset can be found on the [Garbage Classification](https://www.kaggle.com/asdasdasasdas/garbage-classification) page on Kaggle, or downloaded directly through [here](https://www.kaggle.com/asdasdasasdas/garbage-classification/download).