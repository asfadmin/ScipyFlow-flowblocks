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

import gc
import os
import zipfile

def main(zip_path):
    download_path = os.path.split(zip_path)[0]

    # Unzip
    print(f'Extracting {zip_path} to {download_path}')
    print(f'is file zip: {zipfile.is_zipfile(zipfile.ZipFile(zip_path, 'r'))}')

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(download_path)

    os.remove(zip_path)
    gc.collect()

    # So tacky
    return zip_path[0:-4] + ".SAFE"
