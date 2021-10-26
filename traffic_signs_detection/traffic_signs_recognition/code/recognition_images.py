import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt

image_num = '00074'

# Reading image with OpenCV library
# In this way image is opened already as numpy array
image_BGR = cv2.imread(
    f'data/traffic-signs-dataset-in-yolo-format/ts/ts/{image_num}.jpg')
labels = pd.read_csv('data/traffic-signs-preprocessed/label_names.csv')
print(labels.head())
print(labels['SignName'][1])  # Speed limit (30km/h)

# Showing image shape
print('Image shape:', image_BGR.shape)  # tuple of (800, 1360, 3)

# Getting spatial dimension of input image
h, w = image_BGR.shape[:2]  # Slicing from tuple only first two elements

# Showing height an width of image
print('Image height={0} and width={1}'.format(h, w))  # 800 1360

############################################################################

# Reading annotation txt file that has bounding boxes coordinates in YOLO format
with open(f'data/traffic-signs-dataset-in-yolo-format/ts/ts/{image_num}.txt') as f:
    # Preparing list for annotation of BB (bounding boxes)
    lst = []
    for line in f:
        lst += [line.rstrip()]
        print(line)

# Going through all BB
for i in range(len(lst)):
    # Getting current bounding box coordinates, its width and height
    bb_current = lst[i].split()
    x_center, y_center = int(
        float(bb_current[1]) * w), int(float(bb_current[2]) * h)
    box_width, box_height = int(
        float(bb_current[3]) * w), int(float(bb_current[4]) * h)

    # Now, from YOLO data format, we can get top left corner coordinates
    # that are x_min and y_min
    x_min = int(x_center - (box_width / 2))
    y_min = int(y_center - (box_height / 2))

    # Drawing bounding box on the original image
    cv2.rectangle(image_BGR, (x_min, y_min),
                  (x_min + box_width, y_min + box_height), [255, 0, 0], 2)

    # Preparing text with label and confidence for current bounding box
    print(bb_current[0])
    print(labels['SignName'][int(bb_current[0])])

    class_current = 'Class: {}'.format(labels['SignName'][int(bb_current[0])])

    # Putting text with label and confidence on the original image
    cv2.putText(image_BGR, class_current, (x_min, y_min - 5),
                cv2.FONT_HERSHEY_COMPLEX, 0.7, [255, 0, 0], 2)

############################################################################

# Plotting this example
# Setting default size of the plot
plt.rcParams['figure.figsize'] = (15, 15)

# Initializing the plot
fig = plt.figure()

plt.imshow(cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title(f'Image {image_num}.jpg with Traffic Signs', fontsize=18)

# Showing the plot
plt.show()

# Saving the plot
fig.savefig('results3.png')
plt.close()
