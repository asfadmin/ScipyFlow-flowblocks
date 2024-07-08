"""
name: "Orient Image"
requirements:
    - pillow
inputs:
    out_image:
        type: !CustomClass PIL.Image.Image
outputs:
    tiff_info:
        type: !CustomClass PIL.Image.Image
description: "Orients an image"
"""

from PIL import Image, ImageOps

def main (out_image):

    # Mirror data
    out_image_oriented = ImageOps.mirror(out_image)

    # Rotate 180
    out_image_oriented = out_image_oriented.rotate(190, Image.NEAREST, expand = 1)

    return out_image_oriented
