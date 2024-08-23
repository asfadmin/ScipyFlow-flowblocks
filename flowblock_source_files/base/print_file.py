"""
name: "Print File"
requirements:
inputs:
    path:
        type: Str
        user_input: Text
outputs:
    success:
        type: Bool
description: "Downloads a file to the specified path"
"""

def main(path):

    # write file
    contents = ""
    with open(path, "r") as f:
        for line in f:
            contents += line
    print(f'Contents of {path}\n\n{contents}')
    return True
