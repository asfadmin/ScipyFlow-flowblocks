"""
name: "Unpickle"
inputs:
    pickled:
        type: Str
outputs:
    unpickled:
        type: Any
description: "Unpickles a pickle object"
"""

import pickle

def main(pickled) -> any:
    return pickle.loads(pickled)