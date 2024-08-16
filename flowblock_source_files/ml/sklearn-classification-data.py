"""
name: "Make Classification Dataset"
requirements:
    - pandas
    - numpy
    - scikit-learn
inputs:
outputs:
    X:
        type: !CustomClass numpy.ndarray
    y:
        type: !CustomClass numpy.ndarray
description: "Runs an sklearn MLP"
"""

import logging
from sklearn.datasets import make_classification
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def main():
    logging.info(f"Creating random classification dataset")
    X, y = make_classification()
    
    return X, y
