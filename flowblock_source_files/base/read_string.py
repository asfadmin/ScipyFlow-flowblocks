"""
type: "flowblock"
name: "Read User Input String"
inputs:
    input_string:
        type: UserInput
        user_input: True
outputs:
    output_string:
        type: Str
description: "Takes an input string from the user and outputs it."
"""

def main(input_string):
    return input_string
