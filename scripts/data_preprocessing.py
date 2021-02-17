import numpy as np
import os
import struct
from array import array as pyarray
import pickle
import cv2


class CLAHE_GRAY:
    def __init__(self, clipLimit=2.5, tileGridSize=(4, 4)):
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize

    def __call__(self, im):
        img_y = cv2.cvtColor(im, cv2.COLOR_RGB2YCrCb)[:,:,0]
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=self.tileGridSize)
        img_y = clahe.apply(img_y)
        img_output = img_y.reshape(img_y.shape + (1,))
        return img_output


def preprocess(path):
    if not os.path.exists(f"{path}/train_gray.p"):
        for dataset in ['train', 'valid', 'test']:
            with open(f"{path}/{dataset}.p", mode='rb') as f:
                data = pickle.load(f)
                X = data['features']
                y = data['labels']

            # clahe = CLAHE_GRAY()
            for i in range(len(X)):
                # X[i] = clahe(X[i])

                # Histogram normalization in v channel
                hsv = color.rgb2hsv(X[i])
                hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
                img = color.hsv2rgb(hsv)

                # roll color axis to axis 0
                X[i] = np.rollaxis(img, -1)

            X = X[:, :, :, 0]
            with open(f"{path}/{dataset}_gray.p", "wb") as f:
                pickle.dump({"features": X.reshape(
                    X.shape + (1,)), "labels": y}, f)