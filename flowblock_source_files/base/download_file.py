"""
name: "Download File"
requirements:
    - requests
inputs:
    URL:
        type: Str
        default: "/"
        user_input: True

    Path:
        type: Str
        user_input: True
outputs:
    success:
        type: Bool
description: "Downloads a file to the specified path"
"""

import requests

# adding a comment
def main(URL, Path):
    resp = requests.get(URL)
    if resp.status_code != 200:
        return False
    print(resp.data)
