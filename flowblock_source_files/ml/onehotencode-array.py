"""
name: "OneHotEncode Labels"
requirements:
    - numpy
    - scikit-learn
inputs:
    data:
        type: !CustomClass numpy.ndarray
outputs:
    one_hot_encoded:
        type: !CustomClass numpy.ndarray
description: "Returns a onehotencoded version of the array"
"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder

def main(data:np.ndarray)-> np.ndarray:
    enc = OneHotEncoder(sparse_output=False)
    data_one_hot:np.ndarray = enc.fit_transform(data.reshape(-1,1))
    return data_one_hot