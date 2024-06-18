"""
name: "Add"
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
        type: Str
description: "Returns the sum of a and b"
"""

def main(a, b):
    return str(a + b)
