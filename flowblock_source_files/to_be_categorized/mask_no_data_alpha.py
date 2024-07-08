"""
name: "Mask No Data Alpha"
requirements:
    - numpy
    - pillow
inputs:
    out_image:
        type: !CustomClass PIL.Image.Image
outputs:
    image:
        type: !CustomClass PIL.Image.Image
description: "Masks an image with a no-data mask"
"""

import logging
import numpy
from PIL import Image
import gc

def main ( out_image ):
    # Create mask
    logging.info('Generating no-data mask')
    mask = numpy.array(out_image)
    mask = numpy.where(mask > 1, 255, 0)
    mask_image = Image.fromarray(numpy.uint8(mask)).convert('L')

    del mask

    # Remove black borders
    masked_image = out_image.convert('RGBA')
    masked_image.putalpha(mask_image)

    gc.collect()
    return masked_image
