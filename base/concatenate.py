"""
name: "Concatenate"
requirements:
inputs:
    string1:
        type: Str
        default: "Python"
        user_input: True

    string2:
        type: Str
        default: " exists"
        user_input: True
outputs:
    concatenated_string:
        type: Str
description: "Returns the concatenation of two input strings."
"""

def main(string1, string2):
    return string1 + string2
