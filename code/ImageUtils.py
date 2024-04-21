import numpy as np
from matplotlib import pyplot as plt

"""This script implements the functions for data augmentation
and preprocessing.
"""

def parse_record(record, training):
    """Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32].
    """
    image = record.reshape((3, 32, 32))

    return image

def visualize(image, save_name='test.png'):
    """Visualize a single test image.
    
    Args:
        image: An array of shape [3072]
        save_name: An file name to save your visualization.
    
    Returns:
        image: An array of shape [32, 32, 3].
    """
    image = parse_record(image, False)    
    image = np.transpose(image, [1, 2, 0])
    plt.imshow(image)
    plt.savefig(save_name)
    return image