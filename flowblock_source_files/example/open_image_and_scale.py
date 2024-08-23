"""
name: "Open Image and Scale"
requirements:
    - pillow
    - numpy
inputs:
    hh_image_path:
        type: Str
    product_path:
        type: Str
    scale:
        type: Number
        user_input: Text
outputs:
    image:
        type: !CustomClass PIL.Image.Image
description: "Opens an image via an image path and product path, then scales it by a value. Larger numbers mean larger scale factor"
"""

import gc
from PIL import Image
import shutil
import numpy

def main ( hh_image_path: str, product_path: str, scale: int ):

    image = open_image(hh_image_path, product_path)
    imarray = image2array(image)
    del image
    gc.collect()
    imarray = normalize_greys( imarray )
    gc.collect()
    out_image = array2image( imarray )
    del imarray
    gc.collect()
    out_image = scale_image ( out_image, scale )
    return out_image


def open_image (hh_image_path, product_path):
    print(f'Reading product TIFF: {hh_image_path}')
    # Open image as a TIFF

    # This throws "decompression bomb" warnings that can be ignored.
    im = Image.open(hh_image_path)

    # Delete the product to free space
    shutil.rmtree(product_path)

    return im

def image2array (image):
    return numpy.array(image)

def normalize_greys(imarray):
    average_pixel = imarray.mean()
    print(f'Normalizing greyscale to average pixel {average_pixel}')

    # Change to ints
    imarray = numpy.floor(imarray)
    gc.collect()

    imarray = (imarray/average_pixel)*255
    gc.collect()

    return imarray

def array2image( imarray ):
    return Image.fromarray(numpy.uint8(imarray))

def scale_image ( image , scale):
    if type(scale) is not int:
        try:
            scale = int(scale)
        except:
            raise Exception("scale cannot be converted to int")

    print(f'Scaling image by 1:{scale}')
    print(f'image_type: {type(image)}\tscale_type: {type(scale)}')
    scaled_size = [int(x/scale) for x in list(image.size)]
    out_image_small = image.resize( scaled_size )
    del image
    gc.collect()
    return out_image_small
