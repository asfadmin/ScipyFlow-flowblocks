"""
name: "Subtract"
inputs:
    a:
        type: Number
        default: 0
        user_input: True
    b:
        type: Number
        default: 0
        user_input: True
outputs:
    sum:
        type: Number
description: "Returns the difference of a and b"
"""

def main(a, b):
    return a - b
