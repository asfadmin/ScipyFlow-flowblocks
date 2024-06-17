"""
name: "To Integer"
inputs:
    input:
        type: Any
outputs:
    number:
        type: Number
description: "Attempts to convert the input to an integer"
"""

def main(input):
    try:
        return int(input)
    except ValueError:
        raise ValueError
