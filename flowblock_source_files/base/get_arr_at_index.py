"""
name: "Get Value at Index"
requirements:
inputs:
    array:
        type: Sequence

    index:
        type: Number
        default: 0
        user_input: Text
outputs:
    value:
        type: Any
description: "Gets the value at an index of input array"
"""

def main(array, index:int)-> any:
    return array[index]
