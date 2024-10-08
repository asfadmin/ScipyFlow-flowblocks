"""
name: "Score MLPClassifier"
requirements:
    - numpy
    - scikit-learn
inputs:
    X:
        type: !CustomClass numpy.ndarray
    y:
        type: !CustomClass numpy.ndarray
    model: 
        type: !CustomClass sklearn.neural_network._multilayer_perceptron.MLPClassifier
outputs:
    score:
        type: Number
description: "Returns a trained sklearn MLP"
"""
    
import numpy as np
from sklearn.neural_network import MLPClassifier

def main(X:np.ndarray, y:np.ndarray, model:MLPClassifier):
    return model.score(X, y)