"""
type: "flowblock"
name: "Print to Output Text"
inputs:
    input_string:
        type: Str
        user_input: True
description: "Writes the input string to the \"Text Output\" field."
"""

def main(input_string):
    from js import document
    document.getElementById("text").innerText = input_string
