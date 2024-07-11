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
    success:
        type: Bool
description: "Downloads a file to the specified path"
"""

import requests
import os

# adding a comment
def main(URL, path):
    resp = requests.get(URL)
    if resp.status_code != 200:
        return False
    print(resp.text)

    if not os.path.exists(path):
        os.makedirs(path)

    with open(path, "w") as f:
        for line in resp.text:
            f.write(line)
    print(f'File saved to {path}')
