"""
name: "Scale Features"
requirements:
    - numpy
    - scikit-learn
inputs:
    X:
        type: !CustomClass numpy.ndarray
outputs:
    X_scaled:
        type: !CustomClass numpy.ndarray
description: "Returns a trained sklearn MLP"
"""
    
import numpy as np
from sklearn.preprocessing import StandardScaler

def main(X:np.ndarray):
    scalar = StandardScaler().fit(X=X)
    return scalar.transform(X)