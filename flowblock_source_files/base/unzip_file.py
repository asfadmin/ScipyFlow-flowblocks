"""
name: "Unzip"
inputs:
    input_directory:
        type: Str
        user_input: Text
    output_directory:
        type: Str
        user_input: Text
outputs:
    output_directory:
        type: Str
description: "Unzips a zipped folder"
"""

import zipfile

def main(input_directory, output_directory):
    with zipfile.ZipFile(input_directory, 'r') as zip_ref:
        zip_ref.extractall(output_directory)
        extracted_files = zip_ref.namelist()
    return output_directory