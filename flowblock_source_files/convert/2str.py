"""
name: "To String"
inputs:
    input:
        type: Any
outputs:
    string:
        type: Str
description: "Attempts to convert the input to a string"
"""

def main(input):
    try:
        return str(input)
    except ValueError:
        raise ValueError
