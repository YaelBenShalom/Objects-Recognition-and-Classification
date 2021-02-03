import numpy as np
import os
import struct
from array import array as pyarray
import pickle

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

    return features, labels