"""
name: "Weird"
requirements:
inputs:
    string1:
        type: Str
        default: "Gurble"
        user_input: True

    string2:
        type: Str
        user_input: True
outputs:
    concatenated_string:
        type: Str
description: "Does things"
"""

def main(string2, string1):
    return string1 + string2 + " is a thing"
