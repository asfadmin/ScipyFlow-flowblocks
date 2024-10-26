"""
name: "Flatten images"
requirements:
    - numpy
    - pillow
inputs:
    images:
        type: Sequence
outputs:
    flattened_images:
        type: !CustomClass numpy.ndarray
description: "Takes a list of images and returns a numpy array of shape (num_images, num_pixels)"
"""

import numpy as np
from PIL import Image

def main( images:list[Image.Image] ) -> np.ndarray:
    
    width, height = images[0].size
    num_channels = len(images[0].getbands())
    num_pixels = width * height * num_channels
    
    flattened_images:np.ndarray = np.empty((len(images), num_pixels))
    for i, image in enumerate(images):
        flattened_images[i] = np.array(image).flatten()
    
    return flattened_images
