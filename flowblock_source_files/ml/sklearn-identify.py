"""
name: "Create Classifier"
requirements:
    - pandas
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
outputs:
    debug:
        type: !CustomClass sklearn.neural_network._multilayer_perceptron.MLPClassifier
description: "Returns a trained sklearn MLP"
"""

import logging
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def main(
        X,
        y,
        test_size:float = 0.2,
        hidden_layer_sizes:tuple = (100,),
        activation:str = 'relu',
        solver:str = 'adam',
        alpha:float = 0.0001,
        batch_size:int = 'auto',
        learning_rate:str = 'constant',
        learning_rate_init:float = 0.001,
        power_t:float = 0.5,
        max_iter:int = 200,
        shuffle:bool = True,
        random_state:int = None,
        tol:float = 1e-4,
        verbose:bool = False,
        warm_start:bool = False,
        momentum:float = 0.9,
        nesterovs_momentum:bool = True,
        early_stopping:bool = False,
        validation_fraction:float = 0.1,
        beta_1:float = 0.9,
        beta_2:float = 0.999,
        epsilon:float = 1e-8,
        n_iter_no_change = 10,
        max_fun:int = 15000,
        ):
    
    logging.info("Create MLPClassifier")
    
    model = MLPClassifier(
                  hidden_layer_sizes=hidden_layer_sizes,
                  activation=activation,
                  solver=solver,
                  alpha=alpha,
                  batch_size=batch_size,
                  learning_rate=learning_rate,
                  learning_rate_init=learning_rate_init,
                  power_t=power_t,
                  max_iter=max_iter,
                  shuffle=shuffle,
                  random_state=random_state,
                  tol=tol,
                  verbose=verbose,
                  warm_start=warm_start,
                  momentum=momentum,
                  nesterovs_momentum=nesterovs_momentum,
                  early_stopping=early_stopping,
                  validation_fraction=validation_fraction,
                  beta_1=beta_1,
                  beta_2=beta_2,
                  epsilon=epsilon,
                  n_iter_no_change=n_iter_no_change,
                  max_fun=max_fun,
                  )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    model.fit(X_train,y_train)
    
    score = model.score(X_test, y_test)
    
    logging.info(f"Model score: {score}")
    
    return model
