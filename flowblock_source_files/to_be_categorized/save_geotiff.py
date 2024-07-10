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
        user_input: True
outputs:
description: "Saves geotiff to a path"
"""
import logging

def main( image, tiff_info, out_path):
    logging.info('Saving output product')
    image.save(out_path,  tiffinfo=tiff_info)
