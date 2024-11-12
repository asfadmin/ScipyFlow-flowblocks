"""
name: "Model Accuracy"
requirements:
    - numpy
    - scikit-learn
inputs:
    X:
        type: !CustomClass numpy.ndarray
    y_one_hot:
        type: !CustomClass numpy.ndarray
    model: 
        type: !CustomClass sklearn.neural_network._multilayer_perceptron.MLPClassifier
outputs:
    score:
        type: Number
description: "Gets the accuracy of a pretrained sklearn model"
"""
    
import numpy as np
from sklearn.metrics import accuracy_score

def main(X:np.ndarray, y_one_hot:np.ndarray, model):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y_one_hot.argmax(axis=1), y_pred.argmax(axis=1))
    return accuracy