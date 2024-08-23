"""
name: "Uppercase"
requirements:
inputs:
    string:
        type: Str
        default: ""
        user_input: Text
outputs:
    uppercase_string:
        type: Str
description: "Returns the input string put to uppercase"
"""

def main(string):
    return string.upper()
