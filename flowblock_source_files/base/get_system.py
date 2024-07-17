"""
name: "Get System"
requirements:
inputs:
outputs:
    system:
        type: Str
    release:
        type: Str
description: "Returns the system this is being run on"
"""
from platform import system, release

def main():
    return system(), release()
