"""
name: "Semantic Segmentation"
requirements:
    - pandas
    - numpy
    - scipy
    - scikit-learn
inputs:
outputs:
    debug:
        type: Str
description: "A test flowblock to show semantic segmentation in pyodide"
"""

import pandas
from scipy import signal
import numpy as np
import logging
from sklearn.datasets import fetch_openml

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

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
    
class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)
        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient
    
class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input, self.output_shape)
    
    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)
    
# Activation functions
class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))

class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)
        
        super().__init__(sigmoid, sigmoid_prime)

## Loss functions
# MSE
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

# binary_cross_entropy
def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)


# # data specific
# # MNIST
# def preprocess_data(x, y, limit):
#     zero_index = np.where(y == 0)[0][:limit]
#     one_index = np.where(y == 1)[0][:limit]
#     all_indicies = np.hstack((zero_index, one_index))
#     all_indicies = np.random.permutation(all_indicies)
#     x, y = x[all_indicies], y[all_indicies]
#     x = x.reshape(len(x), 1, 28, 28)
#     x = x.astype("float32") / 255
#     y = to_categorical(y)
#     y = y.reshape(len(y), 2, 1)
#     return x, y


## Models

def NN_example():
    # XOR solution data
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

def CNN_example():
    logging.info(f"Starting CNN example")
    logging.info(f"Downloading MNIST dataset")
    # Fetch the MNIST dataset
    mnist = fetch_openml('mnist_784', version=1)

    # Extract features and target labels
    x, y = mnist.data, mnist.target

    # Preprocess data
    

    # Print the shapes
    logging.info(f"Features shape: {x.shape}\tType: {type(x)}")
    logging.info(f"Labels shape: {y.shape}\tType: {type(y)}")


# Execute code
def main():

    # NN_example()
    CNN_example()

    return "Done"

