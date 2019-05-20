import torch
from torch import Tensor
from module import Module
import math

#################### Define the activation models
class Relu(Module):
    # Relu = max(0,X)
    def __init__(self):
        super(Relu, self).__init__()
        self.input = None

    def __repr__(self):
        return "[Relu]\n"

    def forward(self , input):
        self.input = input.clone()
        return self.input.mul(self.input.gt(0).float())

    def backward(self , gradwrtoutput):
        return gradwrtoutput.mul(self.input.gt(0).float())

class Tanh(Module):
    # tanh = (exp(X) - exp(-X))/(exp(X) + exp(-X))
    def __init__(self):
        super(Tanh, self).__init__()
        self.input = None

    def __repr__(self):
        return "[Tanh]\n"

    def forward(self, input):
        self.input = input.clone()
        return self.input.tanh()

    def backward(self, gradwrtoutput):
        #d/dx tanh(x) = 1 - tanh(x)2
        return gradwrtoutput.mul(1 - (self.input.tanh().mul(self.input.tanh())))


class LeakyRelu(Module):
    # LeakyRelu = 0.01*x for x<0
    #             x      for x>= 0
    def __init__(self):
        super(LeakyRelu, self).__init__()
        self.input = None

    def __repr__(self):
        return "[LeakyRelu]\n"

    def forward(self , input):
        self.input = input.clone()
        return self.help_forward(input)
    
    def help_forward(self, x):
        tmp_forward = x
        tmp_forward[tmp_forward<0] = tmp_forward[tmp_forward<0] * 0.01
        return tmp_forward

    def backward(self, gradwrtoutput):
    #         0.01 x<0
    #         1    x>= 0
        tmp_backward = self.input
        tmp_backward[tmp_backward<0] = 0.01
        tmp_backward[tmp_backward>=0] = 1
        return gradwrtoutput * tmp_backward


class Sigmoid(Module):
    # Sigmoid = 1/(1+exp(-X))
    #      x      for x>= 0    
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.input = None

    def __repr__(self):
        return "[Sigmoid]\n"

    def forward(self, input):
        self.input = input.clone()
        tmp = self.input.mul(-1)
        return 1/(1+tmp.exp())

    def backward(self, gradwrtoutput):
    #  sigmoid(1-sigmoid)
        tmp = self.input.mul(-1.0)
        return gradwrtoutput * ((1/(1+tmp.exp()))*(1-(1/(1+tmp.exp()))))

    

