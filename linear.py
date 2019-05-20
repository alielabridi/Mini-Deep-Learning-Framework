import torch
from torch import Tensor
import math
from module import Module

#################### Define the Linear Class for the network 
class Linear(Module):
    def __init__(self, in_features, out_features, bias_init = True):
        super(Linear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.weight_grad = None

        self.bias_init = bias_init
        self.bias_grad = None
        
        self.weight = None
        self.bias = None

    def __repr__(self):
        return '{:<15s}{:^15d}{:^20d}{:^35}{:^35}\n'.\
                    format("[Linear]",self.in_features,\
                    self.out_features,str(self.weight.shape),\
                    str(self.bias.shape) if self.bias_init == True else "None")

    def init_weights(self):
        stdv = 1. / math.sqrt(self.in_features)
        self.weight = Tensor(self.out_features, self.in_features).uniform_(-stdv, stdv)

        self.weight_grad = Tensor(self.out_features, self.in_features)

        if self.bias_init == True:
            stdv = 1. / math.sqrt(self.in_features)
            self.bias = Tensor(self.out_features, 1).uniform_(-stdv, stdv)

            self.bias_grad = Tensor(self.out_features, 1)
        
    def forward(self , input):
        # S = W * X + b
        self.input = input.clone()
        output = self.weight.mm(self.input)
        if self.bias_init == True:
            output.add(self.bias)
        return output
    
    def backward(self , gradwrtoutput):
        if self.bias_init == True:
            # db = dL/ds
            self.bias_grad += gradwrtoutput.sum(dim=1,keepdim=True)
        
        # dW =  dL/ds * X^T 
        self.weight_grad += gradwrtoutput.mm(self.input.t()) 
        
        # dX = dL/ds * W^T
        return self.weight.t().mm(gradwrtoutput)
    
    def param(self):
        return [self.weight, self.bias]

    def param_grad(self):
        return [self.weight_grad, self.bias_grad if self.bias_init == True else None]

    def set_zero_grad(self):
        self.weight_grad.zero_()
        if self.bias_init == True:
            self.bias_grad.zero_()

    def update_param(self, lr):
        # SGD step wt+1 = wt - lr*weight_grad 
        self.weight -= lr*self.weight_grad
        if self.bias_init == True:
            self.bias -= lr*self.bias_grad


