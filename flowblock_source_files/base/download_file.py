"""
name: "Download File"
requirements:
    - requests
inputs:
    URL:
        type: Str
        default: "/"
        user_input: True

    path:
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

def main(URL, path):
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
