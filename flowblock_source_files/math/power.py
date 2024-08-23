"""
name: "Power"
requirements:
    - numpy
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
description: "Returns the a to the power of b"
"""
from numpy import power

def main(a, b):
    return power(a, b)
