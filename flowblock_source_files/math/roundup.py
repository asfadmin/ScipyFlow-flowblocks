"""
name: "Round Up"
requirements:
    - numpy
inputs:
    a:
        type: Number
        default: 0
        user_input: Text
outputs:
    sum:
        type: Number
description: "Rounds the number a up"
"""

from numpy import ceil
def main(a):
    return ceil(a)
