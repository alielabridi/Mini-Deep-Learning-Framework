## This is the framework for a Mini Deep Learning Arch.
import torch
from torch import Tensor
import math
from sequential import Sequential
from activations import Relu, Tanh, LeakyRelu, Sigmoid
from mse import MSE
from linear import Linear
from helpers import generate_disc_set, conv_to_one_hot, compute_nb_errors
import matplotlib
import matplotlib.pyplot as plt

from torch import nn
from torch import optim

def compute_nb_errors_nn(model, data_input, data_target):

    nb_data_errors = 0
    mini_batch_size = 100
    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output, 1)
        for k in range(mini_batch_size):
            if data_target[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors

### Generation of the DATA
train_input, train_target = generate_disc_set(1000)
test_input, test_target = generate_disc_set(1000)
#Standarize Data
mean, std = train_input.mean(), train_input.std()
train_input.sub_(mean).div_(std)
test_input.sub_(mean).div_(std)
#Convert to Labels so that we can train
train_target_hot = conv_to_one_hot(train_target)
test_target_hot = conv_to_one_hot(test_target)


### Build the Network (mini-framework implemented   )
hidden_layers = 3

layers = []
linear = Linear(2, 25, bias_init= True)
layers.append(linear)
layers.append(Relu())
for i in range(hidden_layers-1):
    layers.append(Linear(25, 25, bias_init= True))
    layers.append(Relu())
layers.append(Tanh())
layers.append(Linear(25, 2, bias_init= True))
model = Sequential(layers)
framework_model = Sequential(layers)


def create_deep_model():
    return nn.Sequential(
        nn.Linear(2, 25),
        nn.ReLU(),
        nn.Linear(25, 25),
        nn.ReLU(),
        nn.Linear(25, 25),
        nn.ReLU(),
        nn.Linear(25, 2),
        nn.Tanh(),
    )
pytorch_model = create_deep_model()

pytorch_criterion = nn.MSELoss()
pytorch_optimizer = optim.SGD(pytorch_model.parameters(), lr = 0.05)
nb_epochs = 250

#print model summary
print("model summary:")
print(framework_model)

### Select Parameters to train the framework_model
framework_criterion = MSE()
lr = 0.05
nb_epochs = 250
print_step = 25
mini_batch_size = 100

framework_loss_at_print = []
framework_test_accuracy = []
framework_training_accuracy = []

pytorch_loss_at_print = []
pytorch_test_accuracy = []
pytorch_training_accuracy = []

### Training of the framework_model
for e in range(nb_epochs+1):
    for b in range(0, train_input.size(0), mini_batch_size):

        #Train the mini deep learning framework
        
        ##Forward propagation
        framework_output = framework_model.forward(train_input.narrow(0, b, mini_batch_size).t())

        #Calculate loss
        framework_loss = framework_criterion.forward(framework_output,train_target_hot.narrow(0,b,mini_batch_size).t())
        
        # put to zero weights and bias
        framework_model.zero_grad()
        
        ##Backpropagation
        #Calculate grad of loss
        loss_grad = framework_criterion.backward()

        #Grad of the framework_model
        framework_model.backward(loss_grad)

        #Update parameters
        framework_model.grad_step(lr=lr)

        #Train the model using pytorch NN
        pytorch_output = pytorch_model(train_input.narrow(0, b, mini_batch_size))
        pytorch_loss = pytorch_criterion(pytorch_output, train_target_hot.narrow(0, b, mini_batch_size))
        pytorch_model.zero_grad()
        pytorch_loss.backward()
        pytorch_optimizer.step()

        
    if e % print_step ==0 :
        print(f'Epoc : {e}, Mini Deeplearning Framework Loss: {framework_loss}, Pytorch Loss:{pytorch_loss}')
        #Save loss in vector
        framework_loss_at_print.append(float(framework_loss))
        framework_test_accuracy.append(float(compute_nb_errors(framework_model,test_input, test_target)) / test_target.size(0) * 100)
        print(f'\tMini Deeplearning Framework: tTest Accuracy : {100-framework_test_accuracy[-1]}%')
        framework_training_accuracy.append(float(compute_nb_errors(framework_model,train_input, train_target)) / train_target.size(0) * 100)
        print(f'\tMini Deeplearning Framework: Training Accuracy : {100-framework_training_accuracy[-1]}%')

        pytorch_loss_at_print.append(float(pytorch_loss))
        pytorch_test_accuracy.append(float(compute_nb_errors_nn(pytorch_model,test_input, test_target)) / test_target.size(0) * 100)
        print(f'\tPytorch: Test Accuracy : {100-pytorch_test_accuracy[-1]}%')
        pytorch_training_accuracy.append(float(compute_nb_errors_nn(pytorch_model,train_input, train_target)) / train_target.size(0) * 100)
        print(f'\tPytorch: Training Accuracy : {100-pytorch_training_accuracy[-1]}%')

      

### Printing and plotting Errors and Accuracy
print("\n\n *********************************** \n\n")
print('Mini Deeplearning Framework:\t Number of training errors: ', float(compute_nb_errors(framework_model,train_input, train_target)))
print(f"Mini Deeplearning Framework:\t Accuracy in training: { 100.0 - float(compute_nb_errors(framework_model,train_input, train_target)) / train_target.size(0) * 100} %")
print('Mini Deeplearning Framework:\t Number of test errors: ', float(compute_nb_errors(framework_model,test_input, test_target)))
print(f"Mini Deeplearning Framework:\t Accuracy in test: { 100.0 - float(compute_nb_errors(framework_model,test_input, test_target)) / test_target.size(0) * 100} %")


print("\n *********************************** \n\n")
print('Pytorch:\t Number of training errors: ', float(compute_nb_errors_nn(pytorch_model,train_input, train_target)))
print(f"Pytorch:\t Accuracy in training: { 100.0 - float(compute_nb_errors_nn(pytorch_model,train_input, train_target)) / train_target.size(0) * 100} %")
print('Pytorch:\t Number of test errors: ', float(compute_nb_errors_nn(pytorch_model,test_input, test_target)))
print(f"Pytorch:\t Accuracy in test: { 100.0 - float(compute_nb_errors_nn(pytorch_model,test_input, test_target)) / test_target.size(0) * 100} %")
