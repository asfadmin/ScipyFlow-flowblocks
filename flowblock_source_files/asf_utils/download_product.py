"""
name: "Download Product"
requirements:
    - asf_search==7.0.8
inputs:
    grd_product:
        type: !CustomClass asf_search.ASFProduct.ASFProduct
        user_input: Text

    edl_token:
        type: Str
        user_input: Text

    download_path:
        type: Str
        user_input: Text
        default: "/tmp/data_download"
        
outputs:
    zip_path:
        type: Str
description: "Downloads a product from ASF_SEARCH, returns a file path to the downloaded zip folder"
"""

import asf_search
import os

def main(grd_product, edl_token, download_path='/tmp/data_download'):
    print('Setting up download environment')
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    session = asf_search.ASFSession()

    print(f'Logging into EDL w/ token {edl_token[0:10]}...')
    session.auth_with_token(edl_token)

    print(f'Downloading {grd_product.properties["fileID"]}')
    grd_product.download(path=download_path, session=session)

    fileout = f'{download_path}/{grd_product.properties["sceneName"]}.zip'
    filesizeout = os.stat(fileout).st_size
    print(f'fileout:{fileout}\nfilesizeout:{filesizeout}')

    if download_path == "/":
        return f'/{grd_product.properties["sceneName"]}.zip'

    return f'{download_path}/{grd_product.properties["sceneName"]}.zip'
