"""
name: "Open and Resize Multiple Images"
requirements:
    - pillow
inputs:
    image_paths:
        type: Sequence
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
        default: bilinear
        options:
            - nearest
            - lanczos
            - hamming
            - box
            - bilinear
            - bicubic
outputs:
    resized_images:
        type: Sequence
description: "Opens images and resizes them to dimensions"
"""



from PIL import Image, ImageFile
# I would like the dropdown to allow for non standard types, it may be possible but I do not know

def main (image_paths:list, height:int, width:int, interpolation:str)-> list[Image.Image]:
    ImageFile.LOAD_TRUNCATED_IMAGES=False
    
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
    
    resized_images = []
    for path in image_paths:
        try:
            image = Image.open(path)
            resized_images.append(image.resize((height, width), interp_type))
        except:
            print(f"Could not load image at: {path}")
            pass
    return resized_images
