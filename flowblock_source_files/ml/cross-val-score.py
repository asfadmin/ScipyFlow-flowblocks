"""
name: "Cross Validation Score"
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
    mean:
        type: Number
    standard_deviation:
        type: Number
description: "Returns a trained sklearn MLP"
"""
    
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

def main(X:np.ndarray, y:np.ndarray, model:MLPClassifier):
    scores = cross_val_score(model, X, y)
    return scores.mean(), scores.std()