"""
name: "Output Image"
inputs:
    image:
        type: !CustomClass PIL.Image.Image
        user_input: True
description: "Writes the input PIL image to the \"Image Output\" field."
"""

from js import document

def main(image):
    document.getElementById("img").src = image
