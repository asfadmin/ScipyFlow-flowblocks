"""
name: "Image to Array"
requirements:
    - numpy
inputs:
    image:
        type: !CustomClass PIL.Image.Image
outputs:
    array:
        type: !CustomClass numpy.ndarray
description: "Converts a PIL Image to an numpy array"
"""

import numpy

def main (image):
    return numpy.array(image)
