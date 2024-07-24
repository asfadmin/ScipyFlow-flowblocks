"""
name: "Weird"
requirements:
inputs:
    string2:
        type: Str
        user_input: True

    string1:
        type: Str
        default: "Gurble"
        user_input: True
outputs:
    concatenated_string:
        type: Str
description: "Does things"
"""

def main(string1, string2):
    return string1 + string2 + " is a thing"
