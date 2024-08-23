"""
name: "Lowercase"
requirements:
inputs:
    string:
        type: Str
        default: ""
        user_input: Text
outputs:
    uppercase_string:
        type: Str
description: "Returns the input string put to lowercase"
"""

def main(string):
    return string.lower()
