from layers import NeuralNetwork,DenseLayer
from activations import ReLU_activation,Softmax_activation
from data_creation import spiral_data
import numpy as np
np.random.seed(0)

X,y = spiral_data(100,3)
nn=NeuralNetwork([DenseLayer(3,ReLU_activation()),DenseLayer(3,Softmax_activation())])
print(nn.forward(X))

