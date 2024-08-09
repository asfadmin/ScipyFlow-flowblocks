"""
name: "Semantic Segmentation"
requirements:
    - pandas
    - numpy
    - scikit-learn
inputs:
outputs:
    debug:
        type: Str
description: "A test flowblock to show semantic segmentation in pyodide"
"""

import pandas
import numpy as np
import logging

## Neural Network from scratch
## Made by following tutorials from `The Independent Code`
## https://youtu.be/Lakz2MoHy6o?si=7K1_6qzF9D_uUBTL

# Layers
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input):
        pass

    def backward(self, output_gradient, learning_rate):
        pass

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
    
    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias
    
    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return np.dot(self.weights.T, output_gradient)
    
class Convolutional():
    pass

# Activation functions
class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    
    def backward(self, output_gradient, learning_rate):
        np.multiply(output_gradient, self.activation_prime(self.input))

class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(tanh, tanh_prime)

# Loss functions
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred), 2)

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)



# XOR solution
X = np.reshape([[0,0],[0,1],[1,0],[1,1]], (4,2,1))
Y = np.reshape([[0],[1],[1],[0]],(4,1,1))

network = [
    Dense(2,3),
    Tanh(),
    Dense(3,1),
    Tanh(),
]

epochs = 1000
learning_rate = 0.1

#train
for e in range(epochs):
    error = 0
    for x,y in zip(X,Y):
        # forward
        output = x
        for layer in network:
            output = layer.forward(output)

        # error
        error += mse(y,output)

        # backward
        grad = mse_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)
    
    error /= len(X)
    logging.info('%d/%d, error=%f' % (e + 1, epochs, error))
