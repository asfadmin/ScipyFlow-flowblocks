"""
name: "Orient Image"
requirements:
    - pillow
inputs:
    image:
        type: !CustomClass PIL.Image.Image
outputs:
    image:
        type: !CustomClass PIL.Image.Image
description: "Orients an image"
"""

from PIL import Image, ImageOps

def main (image):

    # Mirror data
    out_image_oriented = ImageOps.mirror(image)

    # Rotate 180
    out_image_oriented = out_image_oriented.rotate(190, Image.NEAREST, expand = 1)

    return out_image_oriented
