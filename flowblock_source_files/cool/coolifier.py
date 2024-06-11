"""
name: "Coolifier"
inputs:
    input_string:
        type: Str
        default: "Python"
        user_input: True
outputs:
    output_string:
        type: Str
description: "Returns the input string concatinated with \" is cool\""
"""

def main(input_string):
    return input_string + " is cool"
