"""
name: "Output Image"
inputs:
    image:
        type: !CustomClass PIL.Image.Image
        user_input: Text
description: "Writes the input PIL image to the \"Image Output\" field."
"""

from js import document
from io import BytesIO
import base64

def main(image):

    # Return image as data
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    data = base64.b64encode(buffered.getvalue()).decode("utf-8")

    document.getElementById("img").src = f"data:image/png;base64,{data}"
