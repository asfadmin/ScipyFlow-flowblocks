"""
name: "Create Classifier"
requirements:
    - scikit-learn
inputs:
    X:
        type: !CustomClass numpy.ndarray
    y:
        type: !CustomClass numpy.ndarray
    hidden_layer_sizes:
        type: Sequence
        default: (100,)
        user_input: Text
    max_iter:
        type: Number
        default: 200
        user_input: Text
    activation:
        type: Str
        default: "relu"
        user_input: Text
    random_state:
        type: Number
        default: None
        user_input: Text
outputs:
    MLPClassifier:
        type: !CustomClass sklearn.neural_network._multilayer_perceptron.MLPClassifier
description: "Returns a trained sklearn MLP"
"""

import logging
from sklearn.neural_network import MLPClassifier
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def main(
        X,
        y,
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
    
    logging.info(f"Reformat inputs")
    logging.info(f"Current Inputs\nhidden_layer_sizes:{hidden_layer_sizes}\t{type(hidden_layer_sizes)}\nmax_iter:{max_iter}\t{type(max_iter)}\nrandom_state:{random_state}\t{type(random_state)}")
    ## Temporary set "None" to None
    ## Remove after scipyflow issue #116 is completed
    from ast import literal_eval
    logging.info(f"literal_eval imported")
    hidden_layer_sizes = tuple(hidden_layer_sizes)
    logging.info(f"hidden_layer_sizes: {hidden_layer_sizes}\ttype:{type(hidden_layer_sizes)}")
    # hidden_layer_sizes = literal_eval(hidden_layer_sizes)
    hidden_layer_sizes = (150, 100, 50)
    logging.info(f"hiddenlayer: {hidden_layer_sizes}\t{type(hidden_layer_sizes)}")
    max_iter = int(max_iter)
    logging.info(f"max_iter: {max_iter}\t{type(max_iter)}")
    if random_state == "None":
        random_state == None
    ## End temporary code
    
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
    
    model.fit(X,y)
    
    return model
