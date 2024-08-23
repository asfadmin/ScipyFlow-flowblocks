"""
name: "Multiply"
inputs:
    a:
        type: Number
        default: 0
        user_input: Text
    b:
        type: Number
        default: 0
        user_input: Text
outputs:
    sum:
        type: Number
description: "Returns the product of a and b"
"""

def main(a, b):
    return a * b
