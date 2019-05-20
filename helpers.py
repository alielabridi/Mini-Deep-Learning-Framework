import torch
from torch import Tensor
import math

# Generate data for the model 
def generate_disc_set(nb):
    input_ = Tensor(nb, 2).uniform_(-1, 1)
    radius = math.sqrt(2/math.pi)
    target = torch.LongTensor(nb)
    target = input_.pow(2).sum(1).sub(2 / math.pi).sign().add(1).div(2).long()
    return input_, target

# Convert the labels to use into the model 
def conv_to_one_hot(labels):
    one_hot = Tensor(labels.shape[0], 2).zero_()
    one_hot.scatter_(1, labels.view(-1, 1), 1.0)
    return one_hot

# Compute Errors 
def compute_nb_errors(model,data_input, data_target):
    nb_data_errors = 0
    output = model.forward(data_input.t())
    predicted_classes = output.max(0)[1]
    return (predicted_classes!=data_target).sum()