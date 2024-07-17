"""
name: "Array to Image"
requirements:
    - numpy
    - pillow
inputs:
    imarray:
        type: !CustomClass numpy.ndarray
outputs:
    image:
        type: !CustomClass PIL.Image.Image
description: "Creates an Image from a numpy array"
"""

import numpy
from PIL import Image

def main( imarray ):
    return Image.fromarray(numpy.uint8(imarray))
