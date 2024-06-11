"""
name: "Cool Num"
inputs:
outputs:
    output_string:
        type: Str
description: "Returns a string of a random number followed by \" cool\""
"""
import random
def main():
    howcool = random.randint(0, 5)
    return howcool + " cool"
