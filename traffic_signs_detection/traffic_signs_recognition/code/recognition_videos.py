import argparse
import os
import numpy as np
import pandas as pd
import cv2
import time
import matplotlib.pyplot as plt
import pickle
from keras.models import load_model

import torch
import torch.nn as nn
# from cnn import BaselineNet
import torchvision.transforms as transforms

# from run_model import run_model, predict


def set_network(config_path, weights_path):
    """
    This function rsets the network and returns the updated network and the names of the
    network's layers.

    Inputs:
        config_path:            path to test configuration.
        weights_path:           path to model's weights.

    Output:
        net:                    the network.
        layers_names_output:    the names of all YOLO v3 layers.
    """
    # Loading trained YOLO v3 weights and cfg configuration file by 'dnn' library from OpenCV
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    # # Set the network to use with GPU
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)

    # Set the network to use with CPU
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Getting names of all YOLO v3 layers
    layers_all = net.getLayerNames()
    layers_names_output = [layers_all[i[0] - 1]
                           for i in net.getUnconnectedOutLayers()]
    # print(f"layers_names_output: {layers_names_output}")

    return net, layers_names_output


def set_output_stream(video, frame_height, frame_width):
    """
    This function defines the output stream of the video.

    Inputs:
        video:                  the processed video.
        frame_width:            the width of the frame.
        frame_height:           the height of the frame.

    Output:
        output_video:           the output stream of the video.
    """

    # Finding the video's fps rate
    fps = video.get(cv2.CAP_PROP_FPS)
    print(f"video FPS: {fps}")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_name = "result.mp4"
    output_video = cv2.VideoWriter(output_video_name, fourcc, fps,
                                   (frame_width, frame_height), True)

    return output_video


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


def get_predictions(net_output, confidence_threshold, nms_threshold,
                    frame_width, frame_height):
    """
    This function finds the network predictions and returns them as bounding boxes,
    confidences, and class numbers lists.

    Inputs:
        net_output:             the network output for the input blob.
        confidence_threshold:   the minimum probability threshold.
        frame_width:            the width of the frame.
        frame_height:           the height of the frame.

    Output:
        bounding_boxes:         list of bounding boxes.
        confidences:            list of confidences in the predictions.
        class_numbers:          list of class numbers.
    """

    # Lists for detected bounding boxes, confidences and class's number
    bounding_boxes = []
    confidences = []
    class_numbers = []

    # Going through all output layers
    for result in net_output:

        # Going through all detections from current output layer
        for detected_objects in result:

            # Getting 80 classes' probabilities for current detected object
            scores = detected_objects[5:]

            # Getting index of the class with the maximum value of probability
            class_current = np.argmax(scores)

            # Getting value of probability for defined class
            confidence_current = scores[class_current]

            # Eliminating weak predictions by minimum probability
            if confidence_current > confidence_threshold:
                try:
                    # Scaling bounding box coordinates to the initial frame size
                    box_current = detected_objects[0:4] * np.array([frame_width, frame_height,
                                                                    frame_width, frame_height])

                    # Getting top left corner coordinates
                    x_center, y_center, box_width, box_height = box_current
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))

                    # Adding results into lists
                    bounding_boxes.append(
                        [x_min, y_min, int(box_width), int(box_height)])
                    confidences.append(float(confidence_current))
                    class_numbers.append(class_current)

                except Exception as e:
                    print(e)

    # Implementing non-maximum suppression of given bounding boxes
    results = cv2.dnn.NMSBoxes(
        bounding_boxes, confidences, confidence_threshold, nms_threshold)

    return results, bounding_boxes, class_numbers


def draw_markers(frame, results, bounding_boxes, scale_factor, mean,
                 class_numbers, model, labels, colors):
    """
    This function draws the bounding boxes and the predicted class on the current frame.

    Inputs:
        frame:                  the current frame.
        results:                the features detected in the current frame.
        bounding_boxes:         the bounding boxes surrounding the detected features.
        scale_factor:           the frame scaling factor.
        mean:                   the mean image.
        class_numbers:          list of class numbers.
        model:                  the trained CNN model.
        labels:                 the labels' names.
        colors:                 colors' list for the bounding boxes.

    Output:
        bounding_boxes:         list of bounding boxes.
        confidences:            list of confidences in the predictions.
        class_numbers:          list of class numbers.
    """

    # Going through indexes of results
    for i in results.flatten():
        # Bounding box coordinates, its width and height
        x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
        box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

        # Cut fragment with Traffic Sign
        frame_ts = frame[y_min:(y_min + int(box_height)),
                         x_min:(x_min + int(box_width)), :]

        if frame_ts.shape[:1] == (0,) or frame_ts.shape[1:2] == (0,):
            pass

        else:
            # Getting preprocessed blob with Traffic Sign of needed shape
            blob_ts = cv2.dnn.blobFromImage(
                frame_ts, scale_factor, size=(32, 32), swapRB=True, crop=False)

            blob_ts[0] = blob_ts[0, :, :, :] - mean["mean_image_rgb"]
            blob_ts = blob_ts.transpose(0, 2, 3, 1)

            # Feeding to the Keras CNN model to get predicted label among 43 classes
            scores = model.predict(blob_ts)
            # print("scores: ", scores)

            # Getting the class with maximum value
            prediction = np.argmax(scores)
            # print("prediction: ", prediction)

            # # Transform tested image
            # blob_ts_tensor = torch.from_numpy(blob_ts)
            # blob_ts_reshaped_tensor = torch.transpose(blob_ts_tensor, 1, 3)

            # # Feeding to the Keras CNN model to get predicted label among 43 classes
            # scores = model(blob_ts_reshaped_tensor)
            # print("scores: ", scores)

            # # Getting the class with maximum value
            # prediction = int(torch.argmax(scores, dim=1))
            # print("prediction: ", prediction)

            # Color for current bounding box
            color_box_current = colors[class_numbers[i]].tolist()

            # Drawing bounding box on the original current frame
            cv2.rectangle(frame, (x_min, y_min), (x_min + box_width, y_min + box_height),
                          color_box_current, 2)

            # Putting text with label and confidence on the original image
            cv2.putText(frame, labels[prediction], (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box_current, 2)

    return frame


def main(args):
    """ Main function of the program
    Inputs:
        args:               the input arguments of the program in the form of a dictionary {"video" : <argument>}.
                            if args exist, <argument> is the input video, else <argument> is None.
    Output: None
    """

    # Define dataset directory
    data_dir = "data"

    # Finding dataset properties
    labels = class_names_fun(data_dir)

    # Loading mean image to use for preprocessing further
    with open("data/mean_image_rgb.pickle", 'rb') as f:
        mean = pickle.load(f, encoding='latin1')

    # Loading trained CNN model to use it later when classifying from 4 groups into one of 43 classes
    model = load_model("model/model-5x5.h5")

    # # Define the device parameters
    # torch.manual_seed(1)
    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")

    # model_path = os.path.abspath("model/model")
    # model = BaselineNet().to(device)
    # model.load_state_dict(torch.load(model_path, map_location=device))

    # Trained weights can be found in the course mentioned above
    weights_path = "weights/signs.weights"
    config_path = "weights/yolov3_ts_test.cfg"

    # Setting the network
    net, layers_names_output = set_network(config_path, weights_path)

    # Generating colors for bounding boxes
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    # Setting confidence and non-maximum suppression thresholds
    confidence_threshold = 0.06
    nms_threshold = 0.08

    if args["video"]:
        # Reading input video
        video = cv2.VideoCapture(args["video"])

        frame_tot = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"video length: {frame_tot} frames")

        # Initializing frame and processing time counters
        current_frame, current_time = 0, 0

        # Capturing frames one-by-one
        ret, frame = video.read()

        # Finding frame size
        frame_height, frame_width = frame.shape[:2]
        print(f"frame_height: {frame_height}\t frame_width: {frame_width}")

        # Set video output stream
        output_video = set_output_stream(video, frame_height, frame_width)

        # Processing the video frame-by-frame
        while video.isOpened():
            start = time.time()

            # Creating blob in current frame
            scale_factor = 1 / 255.0
            size = (416, 416)
            blob = cv2.dnn.blobFromImage(
                frame, scale_factor, size, swapRB=True, crop=False)

            # Forward pass with blob through output layers
            net.setInput(blob)
            net_output = net.forward(layers_names_output)

            end = time.time()
            dt = end - start

            # Updating counters
            current_frame += 1
            current_time += dt

            print(f"Frame: {current_frame}/{frame_tot}\t"
                  f"duration: {dt:.3f} seconds")

            # Get network predictions
            results, bounding_boxes, class_numbers = get_predictions(net_output, confidence_threshold,
                                                                     nms_threshold, frame_width, frame_height)

            # Checking if there is any detected object been left
            if len(results) > 0:
                # # Drawing boxes and predictions on frame
                frame = draw_markers(frame, results, bounding_boxes, scale_factor,
                                     mean, class_numbers, model, labels, colors)

            # Write processed current frame to the file
            output_video.write(frame)

            # Capturing frames one-by-one
            ret, frame = video.read()

            # If the frame was not retrieved
            if not ret:
                print("Stream end. Exiting ...")
                break

        # Release everything if job is finished
        video.release()
        output_video.release()
        cv2.destroyAllWindows()

    print(f"Number of frames: {current_frame}")
    print(f"Total processing time: {current_time:.5f} seconds")
    print(
        f"Average frames per second: {round((current_frame / current_time), 1)}")


if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--video", help="path to the input video")
    args = vars(parser.parse_args())

    # args = {"video": 'input_video/traffic-sign-to-test.mp4'}

    main(args)
