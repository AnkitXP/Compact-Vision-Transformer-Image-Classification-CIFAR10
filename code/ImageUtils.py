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
    image = np.transpose(record.reshape((3, 32, 32)), [1, 2, 0])

    image = preprocess_image(image, training)

    return image


def preprocess_image(image, training):
    """Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [3, 32, 32].
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32]. The processed image.
    """
    if training:
        
        top_padding = np.zeros((32, 4, image.shape[2]))
        side_padding = np.zeros((4, 40, image.shape[2]))
        
        image = np.concatenate([top_padding, image, top_padding], axis=1)
        image = np.concatenate([side_padding, image, side_padding], axis=0)

        width_start, height_start = np.random.randint(0, 8, size=2)
        width_end, height_end = width_start + 32, height_start + 32

        image = image[width_start : width_end, height_start : height_end]

        is_flip = np.random.randint(0, 2, dtype='bool')
        image = np.flip(image, axis=0) if is_flip else image

    mean, std = np.mean(image, axis=(0,1)), np.std(image, axis=(0,1)) 
    image = (image - mean)/std

    return image


def visualize(image, save_name='test.png'):
    """Visualize a single test image.
    
    Args:
        image: An array of shape [3072]
        save_name: An file name to save your visualization.
    
    Returns:
        image: An array of shape [32, 32, 3].
    """
    image = parse_record(image)    
    plt.imshow(image)
    plt.savefig(save_name)
    return image