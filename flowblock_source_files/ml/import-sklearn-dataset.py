"""
name: "Import SKlearn Dataset"
requirements:
    - numpy
    - scikit-learn
inputs:
    dataset_name:
        type: Str
        user_input: Dropdown
        options:
            - "breast_cancer"
            - "diabetes"
            - "digits"
            - "iris"
            - "linnerud"
            - "wine"
    n_class:
        type: Number
        default: None
        user_input: Text
    scaled:
        type: Bool
        default: None
        user_input: Text
outputs:
    X:
        type: !CustomClass numpy.ndarray
    y:
        type: !CustomClass numpy.ndarray
description: "Returns a dataset loaded from sklearn"
"""

from sklearn import datasets
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def main(dataset_name:str, n_class:int=None, scaled:bool=None):
    logging.info(f"Starting to import {dataset_name}")
    
    ## Temporary set "None" to None
    ## Remove after scipyflow issue #116 is completed
    if n_class == "None":
        n_class = None
    if scaled == "None":
        scaled = None
    ## End temporary code
    
    dataset_name = dataset_name.lower()
    kwargs = {
        'return_X_y':True,
        'as_frame':False,
    }
    
    if dataset_name == "breast_cancer":
        return datasets.load_breast_cancer(**kwargs)
    
    elif dataset_name == "diabetes":
        if scaled:
            kwargs[scaled] = scaled
        return datasets.load_diabetes(**kwargs)
    
    elif dataset_name == "digits":
        if n_class:
            kwargs[n_class] = n_class
        return datasets.load_digits(**kwargs)
    
    elif dataset_name == "iris":
        return datasets.load_iris(**kwargs)
    
    elif dataset_name == "linnerud":
        return datasets.load_linnerud(**kwargs)
    
    elif dataset_name == "wine":
        return datasets.load_wine(**kwargs)
    
    else:
        logging.info(f"Cannot find dataset '{dataset_name}'")
        return None, None