"""
name: "Save Geotiff"
requirements:
inputs:
    out_image:
        type: !CustomClass PIL.Image.Image
    tiff_info:
        type: Map
    out_path:
        type: Str
        user_intput: True
outputs:
description: "Saves geotiff to a path"
"""
import logging

def main( out_image, tiff_info, out_path):
    logging.info('Saving output product')
    out_image.save(out_path,  tiffinfo=tiff_info)
