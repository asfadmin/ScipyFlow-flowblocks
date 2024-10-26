"""
name: "Print Directory"
inputs:
    directory:
        type: Str
        user_input: Text
outputs:
description: "Prints contents of directory to console"
"""

import os

def main(directory:str):
    print(os.listdir(directory))