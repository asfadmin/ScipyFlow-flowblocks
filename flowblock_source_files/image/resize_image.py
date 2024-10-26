"""
name: "Resize Image"
requirements:
    - pillow
inputs:
    image:
        type: !CustomClass PIL.Image.Image
    height:
        type: Number
        user_input: Text
        default: 32
    width:
        type: Number
        user_input: Text
        default: 32
    interpolation:
        type: Str
        user_input: Dropdown
        default: nearest
        options:
            - nearest
            - lanczos
            - hamming
            - box
            - bilinear
            - bicubic
outputs:
    array:
        type: !CustomClass PIL.Image.Image
description: "Resizes image to dimensions"
"""

from PIL import Image

def main (image: Image.Image, height:int, width:int, interpolation:int)-> Image.Image:
    interp_type = None
    if interpolation == "nearest":
        interp_type = Image.Resampling.NEAREST
    elif interpolation == "lanczos":
        interp_type = Image.Resampling.LANCZOS
    elif interpolation == "hamming":
        interp_type = Image.Resampling.HAMMING
    elif interpolation == "box":
        interp_type = Image.Resampling.BOX
    elif interpolation == "bilinear":
        interp_type = Image.Resampling.BILINEAR
    elif interpolation == "bicubic":
        interp_type = Image.Resampling.BICUBIC
    return image.resize((height, width), interp_type)
