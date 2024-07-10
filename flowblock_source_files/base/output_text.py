"""
name: "Output Text"
inputs:
    input_string:
        type: Str
        user_input: True
description: "Writes the input string to the \"Text Output\" field."
"""

from js import document

def main(input_string):
    document.getElementById("text").innerText = input_string
