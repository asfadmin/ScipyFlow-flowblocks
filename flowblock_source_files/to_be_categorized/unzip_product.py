"""
name: "Unzip Product"
requirements:
inputs:
    zip_path:
        type: Str
outputs:
    unzipped_path:
        type: Str
description: "Unzips ASF product from a file path"
"""

import logging
import gc
import os
import zipfile

def main(zip_path):
    logging.info(f'Hello is this printed')

    download_path = os.path.split(zip_path)[0]

    # Unzip
    logging.info(f'Extracting {zip_path} to {download_path}')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(download_path)

    os.remove(zip_path)
    gc.collect()

    # So tacky
    return zip_path[0:-4] + ".SAFE"
