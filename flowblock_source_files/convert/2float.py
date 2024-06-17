"""
name: "To Float"
inputs:
    input:
        type: Any
outputs:
    number:
        type: Number
description: "Attempts to convert the input to a float"
"""

def main(input):
    try:
        return float(input)
    except ValueError:
        raise ValueError
