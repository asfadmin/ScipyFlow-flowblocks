"""
name: "Join Arrays"
requirements:
    - numpy
    - scikit-learn
inputs:
    array_1:
        type: !CustomClass numpy.ndarray
    array_2:
        type: !CustomClass numpy.ndarray
outputs:
    joined_array:
        type: !CustomClass numpy.ndarray
description: "Returns the two arrays stacked on top of eachother"
"""

import numpy as np

def main(array_1:np.ndarray, array_2:np.ndarray)-> np.ndarray:
    return np.concatenate((array_1, array_2), axis=0)