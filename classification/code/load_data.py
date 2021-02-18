import os
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
        # # Image in grayscale
        # gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # features[i] = gray.reshape(gray.shape + (1,))

    return features, labels
