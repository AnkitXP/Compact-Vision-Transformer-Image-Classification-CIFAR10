import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms
import torch

import sys


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
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.TrivialAugmentWide(interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.RandomErasing(p=0.1)
        ])

    if training:
        image = train_transform(image)

    # print(image.size())

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