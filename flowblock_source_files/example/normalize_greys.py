"""
name: "Normalize Greys"
requirements:
    - numpy
inputs:
    imarray:
        type: !CustomClass numpy.ndarray
outputs:
    imarray:
        type: !CustomClass numpy.ndarray
description: "Normalizes the greys of a numpy.ndarray"
"""

import gc
import numpy

def main(imarray):
    average_pixel = imarray.mean()
    print(f'Normalizing greyscale to average pixel {average_pixel}')

    # Change to ints
    imarray = numpy.floor(imarray)
    gc.collect()

    imarray = (imarray/average_pixel)*255
    gc.collect()

    return imarray
