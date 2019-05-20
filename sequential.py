import torch
from torch import Tensor
from module import Module
import math

#################### Define the Sequential class of the model 
class Sequential(Module):
    def __init__(self, layers):
        super(Sequential, self).__init__()
        self.layers = layers
        self.init_weights()
    
    def __repr__(self):
        tmp =  120*"-"+"\n"
        tmp +='{:<15s}{:^15s}{:^20}{:^35}{:^35}\n'\
                .format("Layer(Type)","Input(Shape)","Output(shape)","Weight", "Bias")
        tmp += 120*"="+"\n"
        for layer in self.layers:
            tmp += str(layer)
        tmp += 120*"+"+"\n"
        return tmp
        
    def forward(self , input):
        inp = input
        for layer in self.layers:
            out = layer.forward(inp)
            inp = out
        return out

    def backward(self , gradwrtoutput):
        grad = gradwrtoutput
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def _set_zero(self, input):
        return input.fill_(0)

    def init_weights(self):
        for layer in self.layers:
            layer.init_weights()

    def zero_grad(self):
        for layer in self.layers:
            layer.set_zero_grad()
            
    def grad_step(self, lr):
        for layer in self.layers:
            layer.update_param(lr)
