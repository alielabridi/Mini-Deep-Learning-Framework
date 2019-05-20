import torch
from torch import Tensor
from module import Module
import math

#################### Define the Class for the MSE 
class MSE(Module):
    def __init__(self):
        super(MSE, self).__init__()
        self.pred = None
        self.true_val = None
        
    def forward(self, pred, true_val):
        self.pred = pred
        self.true_val = true_val
        return ((pred - true_val).pow(2).sum())/true_val.shape[1]

    def backward(self):
        return 2*(self.pred - self.true_val)/self.true_val.shape[1]