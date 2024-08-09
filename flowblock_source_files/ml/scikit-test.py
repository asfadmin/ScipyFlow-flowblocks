"""
name: "Test ML"
requirements:
    - pandas
    - numpy
    - scikit-learn
inputs:
outputs:
    debug:
        type: Str
description: "Returns the sum of a and b"
"""
    # - ultralytics

import pandas
import numpy
from sklearn.datasets import load_iris
# import ultralytics

def main():
    return "Python ran"