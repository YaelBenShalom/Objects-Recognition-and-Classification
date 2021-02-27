import argparse
import numpy as np
import pandas as pd
import cv2
import time
# from timeit import default_timer as timer
import matplotlib.pyplot as plt
import pickle
# from IPython.display import FileLink
# import os
from keras.models import load_model


def main(args):
    """ Main function of the program
    Inputs:
      args:                 the input arguments of the program in the form of a dictionary {"video" : <argument>}.
                            if args exist, <argument> is the input video, else <argument> is None.
    Output: None
    """
    # Reading csv file with labels' names
    # Loading two columns [0, 1] into Pandas dataFrame
    labels = pd.read_csv("data/traffic-signs-preprocessed/label_names.csv")
    print(labels)

    ############################################################

    # Loading trained CNN model to use it later when classifying from 4 groups into one of 43 classes
    model = load_model("data/model/model-3x3.h5")

    # Loading mean image to use for preprocessing further
    # Opening file for reading in binary mode
    with open("data/traffic-signs-preprocessed/mean_image_rgb.pickle", 'rb') as f:
        mean = pickle.load(f, encoding='latin1')  # dictionary type

    print(mean["mean_image_rgb"].shape)  # (3, 32, 32)

    ###############################################################

    # Trained weights can be found in the course mentioned above
    path_to_weights = "data/traffic-signs-dataset-in-yolo-format/signs.weights"
    path_to_weights_markings = "data/traffic-signs-dataset-in-yolo-format/horizontal.weights"
    path_to_cfg = "data/traffic-signs-dataset-in-yolo-format/yolov3_ts_test.cfg"
    path_to_cfg_markings = "data/traffic-signs-dataset-in-yolo-format/markings_test.cfg"

    # Loading trained YOLO v3 weights and cfg configuration file by 'dnn' library from OpenCV
    network = cv2.dnn.readNetFromDarknet(path_to_cfg, path_to_weights)
    network_markings = cv2.dnn.readNetFromDarknet(path_to_cfg_markings, path_to_weights_markings)

    # To use with GPU
    network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    network.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)

    network_markings.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    network_markings.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)

    ##########################################################

    # Getting names of all YOLO v3 layers
    layers_all = network.getLayerNames()
    layers_names_output = [layers_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]
    print(f"layers_names_output: {layers_names_output}")

    # Getting names of all YOLO v4 layers
    layers_all_markings = network_markings.getLayerNames()
    layers_names_output_markings = [layers_all_markings[i[0] - 1] for i in network_markings.getUnconnectedOutLayers()]
    print(f"layers_names_output_markings: {layers_names_output_markings}")

    ############################################################

    # Minimum probability to eliminate weak detections
    probability_minimum = 0.1

    # Setting threshold to filtering weak bounding boxes by non-maximum suppression
    threshold = 0.1

    # Generating colors for bounding boxes
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
    # colors_markings = np.random.randint(0, 255, size=(1, 3), dtype='uint8')

    #########################################################

    # Reading input video
    # video = cv2.VideoCapture('input_video/70maiMiniDashCam-Dzien.mp4')
    # video = cv2.VideoCapture('input_video/DODRX8W(lusterko)-roadtestwsonecznydzien_podsonce1080p30.mp4')
    # video = cv2.VideoCapture('input_video/traffic-sign-to-test.mp4')
    # video = cv2.VideoCapture('input_video/Video_Recognition.mp4')
    video = cv2.VideoCapture(args["video"])

    frame_tot = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video length: {frame_tot} frames")

    # Writer that will be used to write processed frames
    writer = None

    # Variables for spatial dimensions of the frames
    frame_height, frame_width = None, None

    ########################################################

    # Setting default size of plots
    plt.rcParams["figure.figsize"] = (3, 3)

    # Frame and processing time counters
    current_frame, current_time = 0, 0

    # Capturing frames one-by-one
    ret, frame = video.read()

    # Finding frame size
    frame_height, frame_width = frame.shape[:2]

    # Initializing writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter("result6.mp4", fourcc, 25,
                             (frame_width, frame_height), True)

    # Catching frames in the loop
    while True:

        start = time.time() 

        # Blob from current frame
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # Forward pass with blob through output layers
        network.setInput(blob)
        network_markings.setInput(blob)
        output_from_network = network.forward(layers_names_output)
        output_from_network_markings = network_markings.forward(layers_names_output_markings)

        end = time.time()
        dt = end - start

        # Updating counters
        current_frame += 1
        current_time += dt

        # Spent time for current frame
        print(f"Frame: {current_frame}/{frame_tot}\t"
              f"duration: {dt:.3f} seconds")

        # Lists for detected bounding boxes, confidences and class's number
        bounding_boxes = []
        bounding_boxes_markings = []
        confidences = []
        confidences_markings = []
        class_numbers = []
        class_numbers_markings = []

        # Going through all output layers after feed forward pass
        for result in output_from_network:

            # Going through all detections from current output layer
            for detected_objects in result:

                # Getting 80 classes' probabilities for current detected object
                scores = detected_objects[5:]

                # Getting index of the class with the maximum value of probability
                class_current = np.argmax(scores)

                # Getting value of probability for defined class
                confidence_current = scores[class_current]

                # Eliminating weak predictions by minimum probability

                if confidence_current > probability_minimum:

                    try:
                        # Scaling bounding box coordinates to the initial frame size
                        box_current = detected_objects[0:4] * np.array(
                                      [frame_width, frame_height, frame_width, frame_height])

                        # Getting top left corner coordinates
                        x_center, y_center, box_width, box_height = box_current
                        x_min = int(x_center - (box_width / 2))
                        y_min = int(y_center - (box_height / 2))

                        # Adding results into prepared lists
                        bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                        confidences.append(float(confidence_current))
                        class_numbers.append(class_current)

                    except Exception as e:
                        print(e)

        # Implementing non-maximum suppression of given bounding boxes
        results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)
        results_markings = cv2.dnn.NMSBoxes(bounding_boxes_markings, bounding_boxes_markings,
                                            probability_minimum, threshold)

        # Checking if there is any detected object been left
        if len(results) > 0:
            # Going through indexes of results
            for i in results.flatten():
                # Bounding box coordinates, its width and height
                x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

                # Cut fragment with Traffic Sign
                c_ts = frame[y_min: y_min + int(box_height), x_min: x_min + int(box_width), :]

                if c_ts.shape[:1] == (0,) or c_ts.shape[1:2] == (0,):
                    pass
                else:
                    # Getting preprocessed blob with Traffic Sign of needed shape
                    blob_ts = cv2.dnn.blobFromImage(c_ts, 1 / 255.0, size=(32, 32), swapRB=True, crop=False)
                    blob_ts[0] = blob_ts[0, :, :, :] - mean["mean_image_rgb"]
                    blob_ts = blob_ts.transpose(0, 2, 3, 1)

                    # Feeding to the Keras CNN model to get predicted label among 43 classes
                    scores = model.predict(blob_ts)

                    # Scores is given for image with 43 numbers of predictions for each class
                    # Getting only one class with maximum value
                    prediction = np.argmax(scores)

                    # Color for current bounding box
                    color_box_current = colors[class_numbers[i]].tolist()

                    # Drawing bounding box on the original current frame
                    cv2.rectangle(frame, (x_min, y_min),
                                  (x_min + box_width, y_min + box_height),
                                  color_box_current, 2)

                    # Preparing text with label and confidence for current bounding box
                    # text_box_current = '{}: {:.4f}'.format(labels['SignName'][prediction],
                    #    confidences[i])
                    # text_box_current = '{}'.format(labels["SignName"][prediction])

                    # Putting text with label and confidence on the original image
                    cv2.putText(frame, labels["SignName"][prediction], (x_min, y_min - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box_current, 2)

        # For markings
        for result in output_from_network_markings:
            for detected_objects in result:
                scores = detected_objects[5:]
                class_current = np.argmax(scores)
                confidence_current = scores[class_current]
                if confidence_current > probability_minimum:
                    try:
                        box_current = detected_objects[0:4] * np.array(
                            [frame_width, frame_height, frame_width, frame_height])

                        x_center, y_center, box_width, box_height = box_current
                        x_min = int(x_center - (box_width / 2))
                        y_min = int(y_center - (box_height / 2))

                        bounding_boxes_markings.append([x_min, y_min, int(box_width), int(box_height)])
                        confidences_markings.append(float(confidence_current))
                        class_numbers_markings.append(class_current)
                    except Exception as e:
                        print(e)

        if len(results_markings) > 0:
            for i in results_markings.flatten():
                x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

                cv2.rectangle(frame, (x_min, y_min),
                              (x_min + box_width, y_min + box_height),
                              colors[0].toList(), 2)

        # Write processed current frame to the file
        writer.write(frame)

        # Capturing frames one-by-one
        ret, frame = video.read()

        # If the frame was not retrieved
        if not ret:
            print("Stream end. Exiting ...")
            break

    # Release everything if job is finished
    video.release()
    writer.release()
    cv2.destroyAllWindows()

    ################################################################
    
    print(f"Number of frames: {current_frame}")
    print(f"Total processing time: {current_time:.5f} seconds")
    print(f"Average FPS: {round((current_frame / current_time), 1)}")



if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--video", help="path to the input video")
    args = vars(parser.parse_args())
    main(args)
