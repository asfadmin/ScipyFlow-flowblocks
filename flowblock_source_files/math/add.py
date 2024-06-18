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

def main(a= 0, b=0):
    print("type of a:", type(a),"type of b:",  type(b))
    print("a:", a)
    print("b:", b)
    return str(a + b)
