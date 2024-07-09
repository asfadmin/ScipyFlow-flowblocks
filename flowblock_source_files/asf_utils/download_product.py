"""
name: "Download Product"
requirements:
    - asf_search
inputs:
    grd_product:
        type: !CustomClass asf_search.ASFProduct.ASFProduct
        user_input: True

    edl_token:
        type: Str
        user_input: True

    download_path:
        type: Str
        user_input: True
        default: "/tmp/data_download"
        
outputs:
    zip_path:
        type: Str
description: "Downloads a product from ASF_SEARCH, returns a file path to the downloaded zip folder"
"""

import logging
import asf_search
import os

def main(grd_product, edl_token, download_path='/tmp/data_download'):
    logging.info('Setting up download environment')
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    session = asf_search.ASFSession()

    logging.info(f'Logging into EDL w/ token {edl_token[0:10]}...')
    session.auth_with_token(edl_token)

    logging.info(f'Downloading {grd_product.properties["fileID"]}')
    grd_product.download(path=download_path, session=session)

    return f'{download_path}/{grd_product.properties["sceneName"]}.zip'
