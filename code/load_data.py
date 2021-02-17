import numpy as np
import os
import struct
from array import array as pyarray
import pickle
import cv2

def load_dataset(dataset_name, base_folder='data'):
    """
    This function loads the synthesized data provided in a pickle file in the
    /data directory.
    """

    data_path = os.path.join(base_folder, dataset_name)

    with open(data_path, 'rb') as handle:
        data = pickle.load(handle)

    features = data['features']
    labels = data['labels']

    for i in range(len(features)):
        # Histogram normalization in v channel
        hsv = cv2.cvtColor(features[i], cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        features[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return features, labels


# def load_gray_dataset(path):
#     if not os.path.exists(f"{path}/train_gray.p"):
#         for dataset in ['train', 'valid', 'test']:
#             with open(f"{path}/{dataset}.p", mode='rb') as f:
#                 data = pickle.load(f)
#                 X = data['features']
#                 y = data['labels']

#             # clahe = CLAHE_GRAY()
#             for i in range(len(X)):
#                 # X[i] = clahe(X[i])

#                 # Histogram normalization in v channel
#                 hsv = color.rgb2hsv(X[i])
#                 hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
#                 img = color.hsv2rgb(hsv)

#                 # roll color axis to axis 0
#                 X[i] = np.rollaxis(img, -1)

#             X = X[:, :, :, 0]
#             with open(f"{path}/{dataset}_gray.p", "wb") as f:
#                 pickle.dump({"features": X.reshape(
#                     X.shape + (1,)), "labels": y}, f)