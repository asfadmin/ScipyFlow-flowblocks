"""
name: "Output Text"
inputs:
    input_string:
        type: Str
        user_input: Text
description: "Displays the input_string based on platform"
"""

from platform import system

def main(input_string):
    if system() == "Emscripten":
        from js import document
        document.getElementById("text").innerText = input_string
    else:
        print(input_string)
