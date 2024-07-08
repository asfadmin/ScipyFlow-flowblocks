"""
name: "Get Paths From Product"
requirements:
    - glob
inputs:
    product_path:
        type: Str
outputs:
    hh_image_path:
        type: Str
    hh_metadata_path:
        type: Str
description: "Gets the image path and metadata path from a product path and returns in that order"
"""

import glob
def main(product_path):
    hh_image_path = glob.glob(f'{product_path}/measurement/s1*-*-grd-hh-*.tiff')[0]
    hh_metadata_path = glob.glob(f'{product_path}/annotation/s1*-*-grd-hh-*.xml')[0]

    return hh_image_path, hh_metadata_path
