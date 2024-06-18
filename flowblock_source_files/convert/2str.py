"""
name: "To String"
inputs:
    input:
        type: Any
outputs:
    string:
        type: Str
description: "Attempts to convert the input to a string"
"""

def main(input):
    print(f"2str input {input}")
    try:
        return str(input)
    except ValueError:
        print(f"Error converting to str")
        raise ValueError
