"""
name: "Download File"
requirements:
    - requests
inputs:
    URL:
        type: Str
        default: "/"
        user_input: True

    directory:
        type: Str
        user_input: True
outputs:
    path:
        type: Str
    success:
        type: Bool
description: "Downloads a file to the specified path"
"""

import requests
import os

def main(URL, directory):
    file_name = os.path.split(URL)[1]
    if directory == "/":
        path = f'/{file_name}'
    else:
        path = f'{directory}/{file_name}'

    # get file from URL
    resp = requests.get(URL)
    if resp.status_code != 200:
        return False

    # create path if it does not exist
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    # write file
    with open(path, "w") as f:
        for line in resp.text:
            f.write(line)
    print(f'File saved to {path}')
    return path, True
