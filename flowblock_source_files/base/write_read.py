"""
name: "WriteRead file"
requirements:
inputs:
    path:
        type: Str
        user_input: True
    write_mode:
        type: Bool
        user_input: True
        default: False
    write_contents:
        type: Str
        user_input: True
        default: ""
outputs:
    string:
        type: Str
description: "Writes or reads to a path depending on write_mode, returns path on successful write"
"""
from pathlib import Path

def main(path: str, write_mode: bool, write_contents: str):
    print(f'path:{path}\nwrite_mode:{write_mode}\nwrite_contents:{write_contents}')
    if write_mode:
        print("writing")
        Path(path).write_text(write_contents)
        return path
    
    print("reading")
    return Path(path).read_text()
