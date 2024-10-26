"""
name: "Label Data"
requirements:
    - numpy
inputs:
    data:
        type: !CustomClass numpy.ndarray
    label:
        type: Str
        user_input: Text
outputs:
    X:
        type: !CustomClass numpy.ndarray
    y:
        type: !CustomClass numpy.ndarray
description: "Returns the input data alongside an array of the same length of the given label"
"""

import numpy as np

def main(data:np.ndarray, label:str)-> tuple[np.ndarray, np.ndarray]:
    num_entries = data.shape[0]
    labels = np.full(num_entries, label)
    return data, labels