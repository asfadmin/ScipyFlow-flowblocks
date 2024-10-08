"""
name: "Capitalize"
requirements:
inputs:
    string:
        type: Str
        default: ""
        user_input: Text
outputs:
    uppercase_string:
        type: Str
description: "Capitalizes the input string"
"""

def main(string):
    return string.capitalize()
