"""
name: "Save Geotiff"
requirements:
inputs:
    image:
        type: !CustomClass PIL.Image.Image
    tiff_info:
        type: Map
    out_path:
        type: Str
        user_input: Text
outputs:
description: "Saves geotiff to a path"
"""

def main( image, tiff_info, out_path):
    print('Saving output product')
    image.save(out_path,  tiffinfo=tiff_info)
