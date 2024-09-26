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

from scipy import signal
import pandas as pd
import numpy as np
import logging
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

## Neural Network from scratch
## Made by following tutorials from `The Independent Code`
## https://youtu.be/Lakz2MoHy6o?si=7K1_6qzF9D_uUBTL

class Model:
    def __init__(self, network:list=[]):
        self.network = network
    
    def train(self, x_train, y_train, epochs:int=50, learning_rate:int=0.1):
        epochs = 100
        learning_rate = 0.1
        
        logging.info(f"Start training")
        # train
        for e in range(epochs):
            error = 0
            for x, y in zip(x_train, y_train):
                # forward
                output = x
                for layer in self.network:
                    output = layer.forward(output)
                    
                # error
                error += binary_cross_entropy(y, output)
                
                # backward
                grad = binary_cross_entropy_prime(y, output)
                for layer in reversed(self.network):
                    grad = layer.backward(grad, learning_rate)
            error /= len(x_train)
            logging.info(f"epoch = {e + 1}/{epochs}, error = {error}")
    
    def test(self, x_test, y_test):
        for x, y in zip(x_test, y_test):
            output = x
            for layer in self.network:
                output= layer.forward(output)
            logging.info(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
            
    def save_model(self, file_name:str="model.pkl"):
        with open(file_name, "wb") as file:
            pickle.dump(self.network, file)
            
    def load_model(self, file_name:str=None, file:Path=None):
        if file_name and not file:
            with open(file_name, as)

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
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        # logging.info(f"input_gradient shape:{input_gradient.shape} for DENSE: {input_gradient}")
        return input_gradient
    
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
        # logging.info(f"input_gradient for CONV: {input_gradient}")
        return input_gradient
    
class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input, self.output_shape)
    
    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)
    
## Pooling
# only works with square pools
class Pooling2D(Layer):
    def __init__(self, input_shape: tuple, pool_size=2, strides:int =None, padding='valid', pool_mode="max"):
        self.input_shape = input_shape
        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.strides = pool_size if strides == None else strides
        if padding == "valid":
            self.padding = 0
        elif padding == 'same':
            self.padding = pool_size // 2
        else:
            raise Exception("padding must be either 'valid' or 'same'")
        output_width = (input_shape[2] - pool_size + 2 * self.padding) // self.strides + 1 
        output_height = (input_shape[1] - pool_size + 2 * self.padding) // self.strides + 1 
        output_depth = input_shape[0]
        self.output_shape = (output_depth, output_height, output_width)
        
    def forward(self, input):
        self.padded_input = np.pad(input, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        if self.pool_mode == "max":
            output = np.zeros(self.output_shape)
            # logging.info(f"input shape: {input.shape}\toutput shape: {output.shape}")
            
            horizontal_iterations = (self.input_shape[2] + 2 * self.padding - self.pool_size) // self.strides + 1
            vertical_iterations =  (self.input_shape[1] + 2 * self.padding - self.pool_size) // self.strides + 1
            
            for i in range(0, horizontal_iterations):
                for j in range(0, vertical_iterations):
                    # define window
                    start_x = i * self.strides
                    start_y = j * self.strides
                    end_x = start_x + self.pool_size
                    end_y = start_y + self.pool_size
                    
                    # Extract the region and find the maximum value
                    region = self.padded_input[:, start_x:end_x, start_y:end_y]
                    for k, _ in enumerate(region[0]):
                        max_val = np.max(region[k, :, :])
                        output[k, i, j] = max_val

            return output
        elif self.pool_mode == "average":
            logging.info("average not implemented")
            pass
        else:
            raise Exception("pool_mode must be either 'max' or 'average'")
        
    
    def backward(self, output_gradient, learning_rate):
        # logging.info(f"backprop time")
        if self.pool_mode == "max":
            # Initialize the gradient with zeros
            input_gradient = np.zeros(self.input_shape)
            # logging.info(f"output_gradient: {output_gradient.shape}")
            # logging.info(f"POOL back shape: {input_gradient.shape}")
            
            horizontal_iterations = (self.input_shape[2] + 2 * self.padding - self.pool_size) // self.strides + 1
            vertical_iterations = (self.input_shape[1] + 2 * self.padding - self.pool_size) // self.strides + 1
            # logging.info(f"horiz:{horizontal_iterations}\tvert:{vertical_iterations}")
            
            for i in range(0, horizontal_iterations):
                for j in range(0, vertical_iterations):
                    start_x = i * self.strides
                    start_y = j * self.strides
                    end_x = start_x + self.pool_size
                    end_y = start_y + self.pool_size
                    
                    # Define the region in the padded input
                    # region = self.padded_input[:, start_x:end_x, start_y:end_y]
                    region = output_gradient[:, i, j]
                    # logging.info(f"region shape:{region.shape}")
                    for k in range(0, region.shape[0]):
                        input_gradient[k, start_x:end_x, start_y:end_y] = output_gradient[k, i, j]
                        # logging.info(f"input_gradient:{input_gradient[k, :, :]}")
                    
                    # # Find the max value position
                    # max_pos = np.unravel_index(np.argmax(region, axis=None), region.shape)
                    
                    # # Propagate the gradients
                    # for k in range(0,self.input_shape[0]):
                    #     input_gradient[k, start_x:end_x, start_y:end_y][max_pos] += output_gradient[k, i, j]
            
            return input_gradient
        elif self.pool_mode == "average":
            logging.info("average not implemented")
            return output_gradient
        else:
            raise Exception("pool_mode must be either 'max' or 'average'")
    
class MaxPooling2D(Pooling2D):
    def __init__(self, input_shape: tuple, pool_size=2, strides:int =None, padding='valid'):
        super().__init__(input_shape, pool_size, strides, padding, 'max')
        
## Upsampling
class Upsampling2D(Layer):
    def __init__(self, input_shape: tuple, upsample_dim: tuple = (2,2)):
        self.input_shape = input_shape
        self.upsample_dim = upsample_dim
        self.output_shape = (input_shape[0], input_shape[1] * upsample_dim[0], input_shape[2] * upsample_dim[1])
    
    def forward(self, input):
        # logging.info(f"input shape: {input.shape}")
        self.input = input
        output = np.zeros(self.output_shape)
        
        # depth:int = self.input_shape[0]
        horizontal_iterations:int = self.input_shape[1]
        vertical_iterations:int = self.input_shape[2]
        
        for i in range(0, horizontal_iterations):
            for j in range(0, vertical_iterations):
                start_x = i * self.upsample_dim[0]
                start_y = j * self.upsample_dim[1]
                end_x = start_x + self.upsample_dim[0]
                end_y = start_y + self.upsample_dim[1]
                
                region = input[:, i, j]
                for k in range(0, region.shape[0]):
                    output[k, start_x:end_x, start_y:end_y] = input[k, i, j]
        return output
    
    def backward(self, output_gradient, learning_rate):
        
        input_gradient = np.zeros(self.output_shape)
        
        horizontal_iterations = (self.input_shape[2] + 2 * 0 - self.input_shape[1]) // 1 + 1
        vertical_iterations =  (self.input_shape[1] + 2 * 0 - self.input_shape[2]) // 1 + 1
        
        for i in range(0, horizontal_iterations):
            for j in range(0, vertical_iterations):
                # define window
                start_x = i * self.upsample_dim[0]
                start_y = j * self.upsample_dim[1]
                end_x = start_x + self.upsample_dim[0]
                end_y = start_y + self.upsample_dim[1]
                
                # Extract the region and find the maximum value
                region = output_gradient[:, start_x:end_x, start_y:end_y]
                for k, _ in enumerate(region[0]):
                    max_val = np.max(region[k, :, :])
                    input_gradient[k, i, j] = max_val
        return input_gradient
    
## Activation functions
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
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

## Helper functions
def to_categorical(y: pd.Series):
    # Identify unique categories
    unique_entries = y.unique()
    
    # Create a mapping from category to column index
    category_to_index = {category: idx for idx, category in enumerate(unique_entries)}
    
    # Initialize the output array
    output_arr = np.zeros((len(y), len(unique_entries)))
    
    # Populate the output array
    for i, entry in enumerate(y):
        output_arr[i, category_to_index[entry]] = 1
    
    return output_arr

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
    ## Fetch the MNIST dataset
    mnist = fetch_openml('mnist_784', version=1)

    # Extract features and target labels
    x: np.ndarray = mnist.data.to_numpy()
    y: pd.Series = mnist.target

    ## Preprocess data
    # Get only data for digits 0 and 1
    classes = ['0','1', '2', '3']
    mask = y.isin(classes)
    x = x[mask]
    y = y[mask]
    
    # Reshape inputs
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = to_categorical(y)
    y = y.reshape(len(y), len(classes), 1)
    
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    num_train = 300
    num_test = 300
    x_train = x_train[:num_train]
    x_test = x_test[:num_test]
    y_train = y_train[:num_train]
    y_test = y_test[:num_test]
    
    logging.info(f"x_train shape: {x_train.shape}\ty_train shape:{y_train.shape}")
    
    network = [
        Convolutional((1, 28, 28), kernel_size=3, depth=5),
        Sigmoid(),
        Reshape((5, 26, 26), (5 * 26 * 26, 1)),
        Dense(5 * 26 * 26, 100),
        Sigmoid(),
        Dense(100, len(classes)),
        Sigmoid()
    ]
    
    epochs = 100
    learning_rate = 0.1
    
    logging.info(f"Start training")
    # train
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = x
            for layer in network:
                output = layer.forward(output)
                
            # error
            error += binary_cross_entropy(y, output)
            
            # backward
            grad = binary_cross_entropy_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)
        error /= len(x_train)
        logging.info(f"epoch = {e + 1}/{epochs}, error = {error}")
        
    # test
    for x, y in zip(x_test, y_test):
        output = x
        for layer in network:
            output= layer.forward(output)
        logging.info(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
        
def Semantic_segmentation():
    # classes = [1,2,3,4]
    # network = [
    #     Convolutional((1, 28, 28), kernel_size=3, depth=5),
    #     Sigmoid(),
    #     Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    #     Dense(5 * 26 * 26, 100),
    #     Sigmoid(),
    #     Dense(100, len(classes)),
    #     Sigmoid()
    # ]
    model = Model(network=[
        Convolutional((1, 28, 28), kernel_size=3, depth=5)
    ])
    model.save_model()
    # logging.info(f"{Convolutional((1, 28, 28), kernel_size=3, depth=5).kernels}")
    pass

# Execute code
def main():

    # NN_example()
    # CNN_example()
    Semantic_segmentation()

    return "Done"

main()