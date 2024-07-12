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
from pathlib import Path

def main(URL, directory):
    # create output path
    file_name = os.path.split(URL)[1]
    if directory == "/":
        path = f'/{file_name}'
    else:
        path = f'{directory}/{file_name}'

    # get file from URL
    resp = requests.get(URL)
    print(f'resp code: {resp.status_code}')
    print(f'resp text: {resp.content}')
    if resp.status_code != 200:
        return False

    # create path if it does not exist
    x = os.path.exists(os.path.dirname(path))
    print(f"does path exist:{x}")
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    # write file
    with open(path, "w") as f:
        print("Writing to file")
        for line in resp.content:
            f.write(line)
    print(f'File saved to {path}')
    return path, True
