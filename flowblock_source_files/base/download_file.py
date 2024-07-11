"""
name: "Download File"
requirements:
    - urllib3
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

import urllib3
import urllib3.request

# adding a comment
def main(URL, Path):
    resp = urllib3.request("GET", URL)
    if resp.status != 200:
        return False
    print(resp.data)
