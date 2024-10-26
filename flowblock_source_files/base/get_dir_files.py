"""
name: "Get Directories Files"
requirements:
inputs:
    directory_path:
        type: Str
        user_input: Text
    
    file_extension:
        type: Str
        user_input: Text

outputs:
    file_paths:
        type: Sequence
description: "Returns the concatenation of two input strings."
"""

import os

def main(directory_path:str, file_extension:str=None)-> list:
    
    files = []
    # iterate over each file and subdirectory
    for obj in os.listdir(directory_path):
        # stop if object is not a file
        if os.path.isfile(os.path.join(directory_path, obj)):
            # Filter by file_extension if provided
            if file_extension:
                if obj.endswith(file_extension):
                    files.append(os.path.join(directory_path, obj))
            else:
                files.append(os.path.join(directory_path, obj))
    
    return files