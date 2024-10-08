"""
name: "Run MLPclassifier"
requirements:
    - numpy
    - scikit-learn
inputs:
    model:
        type: !CustomClass sklearn.neural_network._multilayer_perceptron.MLPClassifier
    input_data:
        type: !CustomClass numpy.ndarray
outputs:
    results:
        type: !CustomClass numpy.ndarray
description: "Runs an sklearn MLP"
"""

import numpy as np
from sklearn.neural_network import MLPClassifier

def main(model:MLPClassifier, input_data:np.ndarray):
    results = model.predict(input_data)
    return results
