"""
name: "Split data"
requirements:
    - numpy
    - scikit-learn
inputs:
    X:
        type: !CustomClass numpy.ndarray
    y:
        type: !CustomClass numpy.ndarray
    test_size:
        type: Number
        default: 0.2
        user_input: Text
    random_state:
        type: Number
        default: None
outputs:
    X_train:
        type: !CustomClass numpy.ndarray
    X_test:
        type: !CustomClass numpy.ndarray
    y_train:
        type: !CustomClass numpy.ndarray
    y_test:
        type: !CustomClass numpy.ndarray
description: "Splits the input data into train and test"
"""

import numpy as np
from sklearn.model_selection import train_test_split

def main(X:np.ndarray, y:np.ndarray, test_size:float=0.2, random_state:int=None):
    ## TEMPORARY CODE
    ## Remove after scipyflow issue #116 is completed
    if random_state == "None":
        random_state = None
    ## End temporary code
    return train_test_split(X, y, test_size=test_size, random_state=random_state)