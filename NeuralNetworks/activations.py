import numpy as np
np.random.seed(0)
from abc import ABC,abstractmethod

class activation(ABC):
    @abstractmethod
    def forward(self,input):
        pass
class  ReLU_activation(activation):
    def forward(self,input):
        self.output= np.maximum(0,input)
        return self.output

class Softmax_activation(activation):
    def forward(self,input):
        exp_values = np.exp(input - np.max(input,axis=1,keepdims=True))
        exp = exp_values/ np.sum(exp_values,axis=1,keepdims=True)
        self.output= exp
        return self.output

class Linear_activation(activation):
    def forward(self,input):
        self.output = input
        return self.output


class Sigmoid_activation(activation):
    def forward(self,input):
        self.output = 1/1+np.exp(-input)
        return self.output

