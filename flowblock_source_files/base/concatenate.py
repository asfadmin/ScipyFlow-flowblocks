"""
name: "Concatenate"
requirements:
inputs:
    string1:
        type: Str
        default: "Python"
        user_input: Text

    string2:
        type: Str
        default: "Exists"
        user_input: Text
outputs:
    concatenated_string:
        type: Str
description: "Returns the concatenation of two input strings."
"""

def main(string1, string2):
    return string1 + string2
