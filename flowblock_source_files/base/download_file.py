"""
name: "Download File"
requirements:
    - requests
inputs:
    URL:
        type: Str
        user_input: Text

    directory:
        type: Str
        default: "/home/web_user/"
        user_input: Text
outputs:
    path:
        type: Str
description: "Downloads a file to the specified path"
"""

import requests
import os

def main(URL, directory):
    file_name = os.path.split(URL)[1]
    
    # create output path    
    output_path = os.path.join(directory, file_name)

    # get file from URL
    resp = requests.get(URL)
    print(f"Request to {URL} status code: {resp.status_code}")
    if resp.status_code != 200:
        return None

    # create path if it does not exist
    x = os.path.exists(os.path.dirname(output_path))
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    # write file
    with open(output_path, 'wb') as f:
        f.write(resp.content)
    return output_path
