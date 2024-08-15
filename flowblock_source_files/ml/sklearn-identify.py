"""
name: "Classifier"
requirements:
    - pandas
    - numpy
    - scikit-learn
inputs:
outputs:
    debug:
        type: Str
description: "Runs an sklearn MLP"
"""

from sklearn.neural_network import MLPClassifier

def main():
    pass
