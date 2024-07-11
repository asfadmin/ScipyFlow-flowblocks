"""
name: "WriteRead file"
requirements:
inputs:
    path:
        type: Str
        user_input: True
    write_mode:
        type: Str
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

def main(path, write_mode, write_contents):
    if write_mode:
        Path(path).write_text(write_contents)
        return path
    return Path(path).read_text()
