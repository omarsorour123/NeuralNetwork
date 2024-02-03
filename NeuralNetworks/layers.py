import numpy as np
from activations import Linear_activation

class DenseLayer:
    def __init__(self, n_neurons,activation = Linear_activation()):
        self.n_neurons = n_neurons
        self.weights = None
        self.biases = None
        self.activation = activation
    def __initialize_weights(self, n_inputs):
        self.weights = 0.10 * np.random.randn(n_inputs, self.n_neurons)
        self.biases = np.zeros((1, self.n_neurons))

    def forward(self, input_data):
        if self.weights is None or input_data.shape[1] != self.weights.shape[0]:
            self.__initialize_weights(input_data.shape[1])

        value = np.dot(input_data, self.weights) + self.biases
        self.output = self.activation.forward(value)
        return self.output

class NeuralNetwork:
    def __init__(self,layers):
        self.layers = layers
    def forward(self,n_input):
        input = n_input
        for layer in self.layers:
            input = layer.forward(input)
        return input
