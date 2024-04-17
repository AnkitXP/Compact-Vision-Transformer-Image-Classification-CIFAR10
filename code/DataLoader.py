import os
import pickle
import numpy as np

"""This script implements the functions for reading data.
"""

def load_data(data_dir):
    """Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches
            are stored.

    Returns:
        x_train: An numpy array of shape [50000, 3072].
            (dtype=np.float32)
        y_train: An numpy array of shape [50000,].
            (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072].
            (dtype=np.float32)
        y_test: An numpy array of shape [10000,].
            (dtype=np.int32)
    """
    data = []
    labels = []

    for i in range(1, 6):
        train_path = os.path.join(data_dir, f'data_batch_{i}')

        with open(train_path, 'rb') as train_inputs:
            dataset = pickle.load(train_inputs, encoding='bytes')
            data.append(dataset[b'data'])
            labels.append(dataset[b'labels'])

    x_train = np.concatenate(data, axis = 0)
    y_train = np.concatenate(labels, axis = 0)

    test_path = os.path.join(data_dir, f'test_batch')

    with open(test_path, 'rb') as test_inputs:
        dataset = pickle.load(test_inputs, encoding='bytes')
        x_test = dict[b'data']
        y_test = dict[b'labels']

    return x_train, y_train, x_test, y_test


def load_testing_images(data_dir):
    """Load the images in private testing dataset.

    Args:
        data_dir: A string. The directory where the testing images
        are stored.

    Returns:
        x_test: An numpy array of shape [N, 3072].
            (dtype=np.float32)
    """

    

    return x_test


def train_valid_split(x_train, y_train, train_ratio=0.8):
    """Split the original training data into a new training dataset
    and a validation dataset.

    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        train_ratio: A float number between 0 and 1.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """

    split_index = int(train_ratio * x_train.shape[0])

    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]

    return x_train_new, y_train_new, x_valid, y_valid

